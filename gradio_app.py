# gradio_app.py
import os
import json
import tempfile
import logging
import re
from pathlib import Path
from functools import lru_cache
from typing import Tuple, List, Any, Optional

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Backend modules (the user's modified files) ---
from db_utils import VectorStore, create_embeddings, Document
from mdl_utils import (
    generate_mdl,
    mdl_to_text,
    generate_sql_query,
    get_llm_provider,
)

# -------------------------
# SentenceTransformer loader
# -------------------------
@lru_cache(maxsize=1)
def load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load a SentenceTransformer once (cached).
    Tries to move to GPU if available (torch must be installed).
    Raises RuntimeError on failure (so caller can handle).
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch

        model = SentenceTransformer(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            # Some SentenceTransformer instances accept .to(), others don't.
            model = model.to(device)
        except Exception:
            # ignore if model doesn't support .to()
            pass

        logging.info("Loaded SentenceTransformer on device: %s", device)
        return model
    except Exception as e:
        logging.exception("Failed to load SentenceTransformer: %s", e)
        raise RuntimeError(
            "SentenceTransformer loading failed. Ensure sentence-transformers is installed."
        ) from e


# -------------------------
# Vector store singleton
# -------------------------
@lru_cache(maxsize=1)
def get_vector_store():
    """
    Return a single VectorStore instance used across the app.
    Dimension must match the embedding model used.
    """
    return VectorStore(dimension=384, index_path="faiss_index")


# -------------------------------------------------
# Adapter: Gradio's uploaded file -> file-like object
# -------------------------------------------------
class GradioUploadedFileAdapter:
    """
    Adapter to provide a consistent minimal interface for uploaded files.

    gr_file may be:
      - filepath string (when gradio gives a local temp path, type='filepath')
      - raw bytes (when using binary)
      - an object with read() and .name attribute (older gradio versions)

    Provides:
      - .getbuffer() -> memoryview
      - .getvalue() -> bytes
      - .name -> filename string
      - .path -> path on disk if we were given a filepath string (optional)
    """

    def __init__(self, gr_file):
        from pathlib import Path

        self._bytes: bytes | None = None
        self.path: Optional[str] = None

        if isinstance(gr_file, str):
            # Gradio often returns a local temporary path.
            self.path = gr_file
            self.name = Path(self.path).name
            with open(self.path, "rb") as f:
                self._bytes = f.read()
        elif isinstance(gr_file, (bytes, bytearray)):
            self._bytes = bytes(gr_file)
            self.name = "uploaded_file"
        else:
            # Generic fallback: try .read() and .name
            try:
                self._bytes = gr_file.read()
                self.name = Path(getattr(gr_file, "name", "uploaded_file")).name
            except Exception:
                self._bytes = b""
                self.name = "uploaded_file"

    def getbuffer(self):
        return memoryview(self._bytes)

    def getvalue(self):
        return self._bytes


# -------------------------
# Utilities: temporary save
# -------------------------
def save_temp_file(uploaded_adapter: GradioUploadedFileAdapter) -> str:
    """
    Save uploaded bytes to a file in the system temp dir and return the path.
    """
    tmp_path = os.path.join(tempfile.gettempdir(), uploaded_adapter.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_adapter.getvalue())
    return tmp_path


# -------------------------
# Read uploaded into a DataFrame
# -------------------------
def read_uploaded_to_df(adapted_file: GradioUploadedFileAdapter) -> pd.DataFrame:
    """
    Read an uploaded file (adapter) into a pandas DataFrame.
    Supports CSV and Excel (xlsx/xls).
    Will attempt fallbacks and raise ValueError if parsing fails.
    Removes rows that are entirely NA.
    """
    tmp_path = save_temp_file(adapted_file)
    try:
        lower = tmp_path.lower()
        if lower.endswith(".csv"):
            df = pd.read_csv(tmp_path)
        elif lower.endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
            # modern Excel: use openpyxl
            df = pd.read_excel(tmp_path, engine="openpyxl")
        elif lower.endswith(".xls"):
            df = pd.read_excel(tmp_path, engine="xlrd")
        else:
            # fallback: try CSV first then raise clearer message
            try:
                df = pd.read_csv(tmp_path)
            except Exception:
                raise ValueError(
                    "Unable to parse uploaded file. Ensure it's a valid CSV or Excel file and that openpyxl/xlrd are installed."
                )
        return df.dropna(how="all")
    finally:
        # attempt to clean up the temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# -------------------------
# SQL extraction helper
# -------------------------
def extract_first_select(sql: str) -> str | None:
    """
    Extract the first SELECT statement from generated SQL text.
    Returns the SELECT statement (including trailing ';' if present), or None.
    """
    if not sql:
        return None
    s = re.sub(r"\s+", " ", sql.strip())
    m = re.search(r"(?i)\bselect\b", s)
    if not m:
        return None
    start = m.start()
    sem = s.find(";", start)
    return s[start : sem + 1] if sem != -1 else s[start:]


# -------------------------
# Create collection helpers
# -------------------------
def create_collection_from_upload(gr_file) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Create a collection from an uploaded file.
    Returns (file_hash, mdl_text, error_message).
      - file_hash: unique id derived from bytes+filename
      - mdl_text: textual MDL representation
      - error_message: None on success, otherwise error string
    The function:
      - reads uploaded file into DataFrame
      - generates MDL from df
      - converts MDL to text
      - embeds the MDL text into the vector store (so RAG finds it)
    """
    if gr_file is None:
        return None, None, "No file uploaded"

    adapter = GradioUploadedFileAdapter(gr_file)
    vs = get_vector_store()
    model = load_sentence_transformer()

    try:
        df = read_uploaded_to_df(adapter)
        if df.empty:
            return None, None, "Uploaded file parsed but contains no data."

        file_hash = f"{hash(adapter.getvalue())}_{adapter.name}"
        mdl = generate_mdl(df, dataset_name=file_hash)
        mdl_text = mdl_to_text(mdl)

        # Create embeddings for the MDL only (so MDL is searchable)
        create_embeddings(
            texts=[mdl_text],
            metadatas=[{"type": "mdl", "file_hash": file_hash}],
            ids=[f"{file_hash}_mdl"],
            file_hash=file_hash,
            model=model,
            vector_store=vs,
        )

        return file_hash, mdl_text, None
    except Exception as e:
        logging.exception("Failed to create collection from upload: %s", e)
        return None, None, f"Error processing file: {e}"


# -------------------------
# Helpers to list and load dataset files on disk
# -------------------------
def list_datasets_from_folder(folder: str = "Dataset", limit: int = 5) -> list:
    """
    Return up to `limit` filenames from `folder` sorted alphabetically.
    Accepts CSV and Excel files.
    Returns a list of filenames (not full paths).
    """
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return []

    files = []
    for ext in ("*.csv", "*.xlsx", "*.xls"):
        files.extend(sorted([str(f) for f in p.glob(ext)]))

    files = sorted(files, key=lambda s: Path(s).name.lower())
    return [Path(fp).name for fp in files[:limit]]


def create_collection_from_path(file_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Create collection from a path on disk (same semantics as create_collection_from_upload).
    Returns (file_hash, mdl_text, error_message).
    """
    try:
        if not file_path or not os.path.exists(file_path):
            return None, None, "File not found."

        lower = file_path.lower()
        if lower.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif lower.endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
            df = pd.read_excel(file_path, engine="openpyxl")
        elif lower.endswith(".xls"):
            df = pd.read_excel(file_path, engine="xlrd")
        else:
            # fallback try csv
            df = pd.read_csv(file_path)

        df = df.dropna(how="all")
        if df.empty:
            return None, None, "File parsed but contains no data."

        file_bytes = Path(file_path).read_bytes()
        file_name = Path(file_path).name
        file_hash = f"{hash(file_bytes)}_{file_name}"

        mdl = generate_mdl(df, dataset_name=file_hash)
        mdl_text = mdl_to_text(mdl)

        vs = get_vector_store()
        model = load_sentence_transformer()
        create_embeddings(
            texts=[mdl_text],
            metadatas=[{"type": "mdl", "file_hash": file_hash}],
            ids=[f"{file_hash}_mdl"],
            file_hash=file_hash,
            model=model,
            vector_store=vs,
        )

        return file_hash, mdl_text, None
    except Exception as e:
        logging.exception("Failed to create collection from path: %s", e)
        return None, None, f"Error: {e}"


# -------------------------
# Execute SELECT SQL on a DataFrame
# -------------------------
def execute_sql_on_df(sql: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute a SELECT-only SQL query on a pandas DataFrame using pandasql.
    This is a simplified adapter that replaces the FROM clause with the provided df.
    Only SELECT queries are allowed for safety.
    """
    from pandasql import sqldf

    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")

    sql_clean = sql.strip().rstrip(";")
    sql_repl = sql_clean

    # If no FROM clause detected, append FROM df_clean
    if re.search(r"(?i)\bFROM\s+[" r"\[]?[^\s;]+[" r"\]]?", sql_repl) is None:
        sql_repl = f"{sql_repl} FROM df_clean"
    else:
        # Replace the first FROM <table> with FROM df_clean for safe substitution
        sql_repl = re.sub(r'(?i)\bFROM\s+(["\[]?[^\s;]+["\]]?)', "FROM df_clean", sql_repl, count=1)

    result = sqldf(sql_repl, {"df_clean": df})
    return result


# -------------------------
# LLM calls for charts/insights (DeepSeek)
# -------------------------
def call_deepseek_for_charts(prompt_text: str, available_columns: List[str], sample_data: List[dict]) -> Optional[str]:
    """
    Ask the LLM (via get_llm_provider()) for up to 2 chart specifications.
    System prompt forces the model to return ONLY a JSON array describing charts.
    Returns the raw LLM string (expected JSON array), or None on failure.
    """
    try:
        llm = get_llm_provider()
        system_msg = (
            "You are a data visualization expert that creates Plotly charts. "
            "Return ONLY a JSON array of up to 2 chart objects. "
            "Each chart object examples:\n"
            '{"type":"bar","title":"...","x":["A","B"],"y":[10,20],"color":"#4285F4"}\n'
            'For pie: {"type":"pie","title":"...","labels":["A"],"values":[10]}\n'
            "Use only categories/values present in the provided sample_data."
        )
        user_msg = (
            f"User query: {prompt_text}\n"
            f"Available columns: {available_columns}\n"
            f"Sample data (first rows): {json.dumps(sample_data, default=str)}\n\n"
            "Return only valid JSON as described."
        )
        prompt = system_msg + "\n\n" + user_msg
        return llm.generate(prompt)
    except Exception as e:
        logging.exception("Error calling DeepSeek for charts: %s", e)
        return None


def call_deepseek_for_insights(prompt_text: str, sample_data: List[dict], generated_sql: Optional[str]) -> Optional[str]:
    """
    Ask the LLM for 2-3 sentence insights about the data.
    Returns plain text insight or None on failure.
    """
    try:
        llm = get_llm_provider()
        system_msg = (
            "You are a data analyst. Write ONLY 2-3 sentences focusing on the most important finding. "
            "Include specific numbers/percentages when possible. Return the insight as plain text."
        )
        user_msg = (
            f"User query: {prompt_text}\n"
            f"Sample data (top rows): {json.dumps(sample_data, default=str)}\n"
            f"SQL used: {generated_sql if generated_sql else 'None'}\n"
        )
        prompt = system_msg + "\n\n" + user_msg
        return llm.generate(prompt)
    except Exception as e:
        logging.exception("Error calling DeepSeek for insights: %s", e)
        return None


# -------------------------
# Convert LLM JSON -> Plotly figures
# -------------------------
def parse_chart_json_and_make_figs(chart_json_str: str) -> List[Any]:
    """
    Parse the LLM-returned JSON (string) describing charts and convert them to Plotly figures.
    Returns a list of Plotly figure objects (possibly empty if parse/build fails).
    The LLM is expected to return a JSON list of up to 2 chart config objects.
    """
    figs: List[Any] = []
    if not chart_json_str:
        return figs

    clean = chart_json_str.strip()

    # Strip markdown fences if present
    if clean.startswith("```json"):
        clean = clean[len("```json") :].strip()
    if clean.startswith("```"):
        clean = clean[3:].strip()
    if clean.endswith("```"):
        clean = clean[:-3].strip()

    # Try to extract first JSON array in text
    try:
        m = re.search(r"(\[.*\])", clean, re.DOTALL)
        json_text = m.group(1).strip() if m else clean
    except Exception:
        json_text = clean

    try:
        parsed = json.loads(json_text)
    except Exception:
        logging.exception("Failed to parse chart JSON from LLM. Attempting fallback extraction.")
        try:
            m2 = re.search(r"(\[.*\])", chart_json_str, re.DOTALL)
            if not m2:
                return figs
            parsed = json.loads(m2.group(1))
        except Exception:
            logging.exception("Fallback JSON parse failed. Raw chart_json_str: %s", chart_json_str[:500])
            return figs

    if not isinstance(parsed, list):
        logging.warning("Parsed chart JSON is not a list; got %s", type(parsed))
        return figs

    # Helper: sanitize colors to either hex or simple name, else None
    def sanitize_color(c):
        if not c:
            return None
        c = str(c).strip()
        if re.match(r"^#([A-Fa-f0-9]{6})$", c):
            return c
        if re.match(r"^[a-zA-Z]+$", c):
            return c
        return None

    for cfg in parsed[:2]:
        if not isinstance(cfg, dict):
            continue
        ctype = cfg.get("type", "bar")
        title = cfg.get("title", "")
        try:
            if ctype == "pie":
                labels = list(cfg.get("labels", []))
                values = list(cfg.get("values", []))
                if not labels or not values or len(labels) != len(values):
                    logging.warning("Invalid pie chart config (labels/values mismatch): %s", cfg)
                    continue
                fig = px.pie(names=labels, values=values, title=title)
                color = sanitize_color(cfg.get("color"))
                if color:
                    fig.update_traces(marker=dict(colors=[color] * len(labels)))
                figs.append(fig)
            else:
                x = list(cfg.get("x", []))
                y = list(cfg.get("y", []))
                if not x or not y or len(x) != len(y):
                    logging.warning("Invalid chart config x/y mismatch or empty: %s", cfg)
                    continue

                x_label = cfg.get("x_label", "")
                y_label = cfg.get("y_label", "")

                if ctype == "line":
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=y_label or "value"))
                    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
                    figs.append(fig)
                elif ctype == "scatter":
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=y_label or "value"))
                    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
                    figs.append(fig)
                else:
                    # default to bar
                    fig = px.bar(x=x, y=y, labels={"x": x_label, "y": y_label}, title=title)
                    color = sanitize_color(cfg.get("color"))
                    if color:
                        fig.update_traces(marker_color=color)
                    figs.append(fig)
        except Exception:
            logging.exception("Error building figure from config: %s", cfg)
            continue

    return figs


# ========================
# Gradio Handlers
# ========================
def upload_file_and_create_collection(gr_file) -> Tuple[str, str, str]:
    """
    Called when a file is uploaded via the file component.
    Returns a triplet:
      - status text (user facing)
      - mdl_text (string)
      - file_hash (string)
    """
    file_hash, mdl_text, err = create_collection_from_upload(gr_file)
    if err:
        return "❌ " + err, "", ""
    return f"✅ Uploaded and MDL embedded (file_hash: {file_hash})", mdl_text or "", file_hash or ""


def on_choose_dataset(dataset_name: str):
    """
    When the user selects a sample dataset from the Dataset/ folder.
    Returns values in this exact order (to match the UI outputs):
      - status (gr.update(...))
      - raw_mdl (string)
      - file_hash_state (string)
      - file_in cleared (gr.update(value=None))
      - preview_df (DataFrame)
    """
    # If no selection -> reset UI
    if not dataset_name:
        return (
            gr.update(value="No dataset selected.", elem_classes="status-warn"),
            "",
            "",
            gr.update(value=None),
            pd.DataFrame(),
        )

    dataset_folder = Path("Dataset")
    candidates = list(sorted(dataset_folder.glob("*"), key=lambda p: p.name.lower()))
    chosen_path = None
    for p in candidates:
        if p.name == dataset_name:
            chosen_path = str(p)
            break

    if not chosen_path:
        return (
            gr.update(value=f"Could not find {dataset_name} in Dataset/", elem_classes="status-error"),
            "",
            "",
            gr.update(value=None),
            pd.DataFrame(),
        )

    # Create collection + MDL (may be slow)
    file_hash, mdl_text, err = create_collection_from_path(chosen_path)
    if err:
        return (
            gr.update(value=f"❌ {err}", elem_classes="status-error"),
            "",
            "",
            gr.update(value=None),
            pd.DataFrame(),
        )

    # Prepare a preview (first 5 rows)
    try:
        if chosen_path.lower().endswith(".csv"):
            preview_df = pd.read_csv(chosen_path).head(5)
        else:
            preview_df = pd.read_excel(chosen_path, engine="openpyxl").head(5)
    except Exception:
        preview_df = pd.DataFrame()

    status_update = gr.update(
        value=f"✅ Loaded dataset '{dataset_name}' (file_hash: {file_hash})", elem_classes="status-ok"
    )

    # Return in the exact order the UI expects:
    # status (gr.update), raw_mdl (string), file_hash (string), file_in cleared (gr.update), preview_df (DataFrame)
    return status_update, (mdl_text or ""), (file_hash or ""), gr.update(value=None), preview_df


def generate_charts_and_insights(gr_file, file_hash, user_prompt) -> Tuple[Any, str, str, str]:
    """
    Main function that generates charts and insights.
    Returns: (figure_or_None, insights_text, debug/status_text, generated_sql_or_empty)
    """
    status = ""
    generated_sql = ""

    try:
        # Validate inputs
        if gr_file is None and not file_hash:
            return None, "No data. Upload a file first.", "No file provided.", ""

        # If gr_file is None but file_hash present, attempt to map to a file under Dataset/
        if gr_file is None and file_hash:
            for p in Path("Dataset").glob("*"):
                if p.name in str(file_hash):
                    gr_file = str(p)
                    logging.info("Mapped file_hash to dataset path: %s", gr_file)
                    break
            if gr_file is None:
                logging.warning("generate_charts_and_insights: gr_file is None but file_hash=%s", file_hash)

        # Adapt file for reading
        adapter = GradioUploadedFileAdapter(gr_file) if gr_file is not None else None
        if adapter is None:
            return None, "No file available to read (uploader empty). Select or upload a file.", "NO_FILE", ""

        df = read_uploaded_to_df(adapter)
        if df is None or df.empty:
            return None, "Uploaded file empty or couldn't parse", "Parsed DataFrame empty", ""

        logging.info("Starting chart generation for dataset rows=%s columns=%s", len(df), list(df.columns)[:10])

        # Load model for embedding/querying
        model = load_sentence_transformer()
        vs = get_vector_store()

        # Encode user prompt and run similarity search to detect whether top doc is the MDL
        query_emb = model.encode([user_prompt], convert_to_numpy=True)[0]
        similar_docs = vs.similarity_search(query_embedding=query_emb, k=10, file_hash=file_hash)
        top_is_mdl = False
        if similar_docs:
            top = similar_docs[0]
            top_meta = top.get("metadata", {}) or {}
            top_is_mdl = top_meta.get("type") == "mdl" or (top.get("id", "").endswith("_mdl"))
        logging.info("RAG: top_is_mdl=%s, top_docs_found=%d", top_is_mdl, len(similar_docs or []))

        query_result_df = None

        if top_is_mdl:
            # Generate SQL using MDL and run it
            try:
                mdl = generate_mdl(df, dataset_name=file_hash)
                generated_sql = generate_sql_query(natural_language_query=user_prompt, mdl=mdl)
                logging.info("Generated SQL (raw): %s", generated_sql)
                safe_sql = extract_first_select(generated_sql)
                logging.info("Generated SQL (safe): %s", safe_sql)

                if isinstance(generated_sql, str) and generated_sql.strip().upper() == "INVALID QUERY":
                    return None, "The query appears unrelated to this dataset (INVALID QUERY).", generated_sql, generated_sql

                if not safe_sql:
                    return None, "Generated SQL is not a SELECT query. Try rephrasing.", generated_sql, generated_sql

                query_result_df = execute_sql_on_df(safe_sql, df)
                if query_result_df is None or query_result_df.empty:
                    query_result_df = df.head(100)

            except Exception as e:
                logging.exception("SQL generation/execution failed: %s", e)
                query_result_df = df.head(100)
        else:
            # Fallback: either try to extract row-like metadata from similar_docs or use a sample
            if similar_docs:
                rows = []
                for d in similar_docs:
                    meta = d.get("metadata", {}) or {}
                    # if metadata keys look like df columns, consider them row-like
                    if isinstance(meta, dict) and set(meta.keys()) & set(df.columns):
                        rows.append(meta)
                if rows:
                    query_result_df = pd.DataFrame(rows)
                else:
                    query_result_df = df.head(100)
            else:
                query_result_df = df.head(100)

        # Prepare sample data for the LLM
        sample_data = query_result_df.head(12).to_dict(orient="records")

        # Ask the LLM for chart JSON
        chart_json = call_deepseek_for_charts(user_prompt, list(df.columns), sample_data)
        logging.info("LLM returned chart JSON (truncated): %s", (chart_json or "")[:500])

        if not chart_json:
            return None, "LLM returned no chart configuration. Try rephrasing.", generated_sql or "No SQL", generated_sql or ""

        figs = parse_chart_json_and_make_figs(chart_json)
        if not figs:
            return None, "Failed to parse chart config from LLM.", generated_sql or "No SQL", generated_sql or ""

        # Generate insights
        insights = call_deepseek_for_insights(user_prompt, sample_data, generated_sql)
        if not insights:
            insights = "Unable to generate insights."

        # Return only the first figure (gr.Plot expects a single figure object)
        first_fig = figs[0]
        return first_fig, insights, "OK", (generated_sql or "")

    except Exception as e:
        logging.exception("Unexpected error in generate_charts_and_insights: %s", e)
        return None, f"Error: {e}", "EXCEPTION", ""


# ========================
# Gradio UI Construction
# ========================
def build_ui():
    css = r"""
    /* Status boxes */
    .status-ok textarea { background-color: #e6f4ea !important; color: #1e4620 !important; border: 1px solid #34a853 !important; }
    .status-error textarea { background-color: #fce8e6 !important; color: #b31412 !important; border: 1px solid #ea4335 !important; }
    .status-warn textarea { background-color: #fff4e5 !important; color: #663c00 !important; border: 1px solid #ffa726 !important; }

    /* Warm orange button (less bright) */
    .btn-warm-orange button { background-color: #ff914d !important; color: #ffffff !important; border-radius: 6px !important; border: none !important; }
    .btn-warm-orange button:hover { filter: brightness(0.95); }

    /* Make accordion text area monospace a bit smaller */
    .gr-accordion .gr-textbox textarea { font-family: monospace; font-size: 12px; }

    /* small card-like box for dataset dropdown */
    .dataset-card { background: transparent; padding: 6px 4px; border-radius: 6px; margin-top: 6px; border: 1px solid rgba(0,0,0,0.06); }
    """

    with gr.Blocks(title="Agent Graphiq - Gradio", css=css) as demo:
        gr.Markdown("# Agent Graphiq — AI Chart Generator")

        with gr.Row():
            # LEFT COLUMN → Upload, Dataset dropdown, Status, MDL accordion, SQL accordion
            with gr.Column(scale=1):
                file_in = gr.File(
                    label="Upload CSV or Excel",
                    file_count="single",
                    type="filepath",
                )

                dataset_choices = list_datasets_from_folder("Dataset", limit=5)
                with gr.Group(elem_classes="dataset-card"):
                    gr.Markdown("**Or pick a sample dataset**")
                    dataset_dd = gr.Dropdown(
                        choices=[""] + dataset_choices,
                        value="",
                        label="",
                        info="Select a sample dataset.",
                    )
                    gr.Markdown("_Selecting a sample dataset will clear the file uploader._")

                status = gr.Textbox(label="Status / Debug", interactive=False, elem_classes="status-warn", value="No file uploaded")

                with gr.Accordion("📑 Dataset MDL (Schema)", open=False):
                    raw_mdl = gr.Textbox(label="MDL", interactive=False, lines=15, value="")

                with gr.Accordion("🧾 Generated SQL", open=False):
                    raw_sql = gr.Textbox(label="SQL (first SELECT shown)", interactive=False, lines=8, value="")

            # RIGHT COLUMN → Preview, Query, Buttons, Charts, Insights
            with gr.Column(scale=2):
                preview_df = gr.Dataframe(value=pd.DataFrame(), label="📊 Dataset Preview (first 5 rows)", interactive=False)

                user_prompt = gr.Textbox(label="Enter your query", placeholder="e.g. Plot engagement by country")

                generate_btn = gr.Button("Generate Charts", elem_classes="btn-warm-orange")

                plots = gr.Plot(label="Charts")

                insights_out = gr.Textbox(label="Data Insights", interactive=False, value="")

                # state variable to store file hash returned by create_collection
                file_hash_state = gr.State("")

                # Upload handler wrapper
                def _on_upload_wrapper(file):
                    try:
                        file_hash, mdl_text, err = create_collection_from_upload(file)
                        if err:
                            return gr.update(value=f"❌ {err}", elem_classes="status-error"), "", "", pd.DataFrame(), "", ""

                        try:
                            adapter = GradioUploadedFileAdapter(file)
                            df = read_uploaded_to_df(adapter)
                            preview = df.head(5)
                        except Exception:
                            preview = pd.DataFrame()

                        status_text = f"✅ Uploaded and MDL embedded (file_hash: {file_hash})"
                        return gr.update(value=status_text, elem_classes="status-ok"), mdl_text or "", file_hash or "", preview, "", ""
                    except Exception as e:
                        logging.exception("Upload wrapper error: %s", e)
                        return gr.update(value="❌ Upload failed", elem_classes="status-error"), "", "", pd.DataFrame(), "", ""

                # Wire the upload: reset dropdown and clear raw_sql on success
                file_in.change(
                    fn=_on_upload_wrapper,
                    inputs=[file_in],
                    outputs=[status, raw_mdl, file_hash_state, preview_df, dataset_dd, raw_sql],
                )

                # Wire dataset dropdown selection
                dataset_dd.change(
                    fn=on_choose_dataset,
                    inputs=[dataset_dd],
                    outputs=[status, raw_mdl, file_hash_state, file_in, preview_df],
                )

                # Chart + insight generation click handler
                def _on_generate(file, file_hash, prompt):
                    fig, insights, dbg, gen_sql = generate_charts_and_insights(file, file_hash, prompt)
                    if fig is None:
                        # dbg likely contains an error message
                        return None, insights, gr.update(value=f"❌ {dbg}", elem_classes="status-error"), gr.update(value=(gen_sql or ""))
                    # success
                    return fig, insights, gr.update(value="✅ OK", elem_classes="status-ok"), gr.update(value=(gen_sql or ""))

                generate_btn.click(
                    fn=_on_generate,
                    inputs=[file_in, file_hash_state, user_prompt],
                    outputs=[plots, insights_out, status, raw_sql],
                )

    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Warm-up heavy resources at startup to speed first user actions.
    try:
        logging.info("Warming up SentenceTransformer and vector store (startup) …")
        _ = load_sentence_transformer()
        _ = get_vector_store()
    except Exception as e:
        logging.exception("Startup warm-up failed: %s", e)

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
