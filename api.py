# api.py

import os
import json
import logging
import base64
import io
import time
import uuid
import re
import pandas as pd
import plotly.express as px

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load env
load_dotenv()

# Import existing utils
from db_utils import VectorStore, Document
from mdl_utils import (
    DatasetMDL,
    generate_mdl,
    mdl_to_text,
    load_dataframe_from_file,
    get_llm_provider,
    generate_sql_query,
    clean_sql,
    classify_user_message,
)

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log")],  # file handler
)

# module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# console handler
console_hdl = logging.StreamHandler()
console_hdl.setLevel(logging.DEBUG)
console_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
console_hdl.setFormatter(console_fmt)

if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(console_hdl)

# also for uvicorn loggers
for uv_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    ul = logging.getLogger(uv_name)
    ul.setLevel(logging.DEBUG)
    if not any(isinstance(h, logging.StreamHandler) for h in ul.handlers):
        ul.addHandler(console_hdl)

logger.propagate = False

# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
app = FastAPI(title="GraphIQ Agent API", version="2.0")

# CORS
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------
# Globals
# ----------------------------------------------------------------------------
DIMENSION = 384
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")

VECTOR_STORE = VectorStore(dimension=DIMENSION, index_path=FAISS_INDEX_PATH)
EMBEDDING_MODEL_INSTANCE = SentenceTransformer(EMBEDDING_MODEL)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-memory datasets {dataset_name: DataFrame}
DATASETS: Dict[str, pd.DataFrame] = {}

# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class QueryInput(BaseModel):
    message: str
    sender: str
    dataset_name: str
    top_k: Optional[int] = 3
    score_threshold: Optional[float] = 0.7


class QueryResponse(BaseModel):
    summary: Optional[str] = None
    chart_image: Optional[str] = None
    plotly_json: Optional[str] = None
    error: Optional[str] = None

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def df_to_chart_base64(df: pd.DataFrame):
    """Generate a simple chart from first 2 columns and return (base64 PNG, plotly_json)."""
    try:
        if df.shape[0] == 0:
            logger.debug("df_to_chart_base64: empty dataframe, skipping chart")
            return None, None

        if df.shape[1] >= 2:
            x_col, y_col = df.columns[:2]
            fig = px.bar(df, x=x_col, y=y_col, title="Auto-generated chart")
        else:
            fig = px.bar(df)

        try:
            fig_json = fig.to_json()
        except Exception as e:
            logger.exception(f"Failed to serialize plotly fig to json: {e}")
            fig_json = None

        img_base64 = None
        try:
            img_bytes = fig.to_image(format="png")
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            img_base64 = f"data:image/png;base64,{img_base64}"
        except Exception as e:
            logger.exception(f"Plotly PNG export failed (kaleido missing?): {e}")
            img_base64 = None

        return img_base64, fig_json

    except Exception as e:
        logger.exception(f"Plotly chart generation failed: {e}")
        return None, None


def summarize_result(df: pd.DataFrame, query: str) -> str:
    """Generate a short text summary using LLM."""
    try:
        llm = get_llm_provider()
        preview = df.head(10).to_csv(index=False)
        prompt = f"""
You are a data analyst. The user asked: {query}

Here are the first 10 rows of the query result:

{preview}

Provide a short natural language summary of the result (2-3 sentences).
"""
        return llm.generate(prompt).strip()
    except Exception as e:
        logger.exception(f"Summary generation failed: {e}")
        return "Summary generation failed."


def sanitize_sql_from_llm(raw: str) -> str:
    """Clean SQL text from LLM."""
    if not raw:
        return raw

    s = raw
    s = re.sub(r"```(?:\s*sql)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```", "", s)
    s = re.sub(r"~~~(?:\s*sql)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*~~~", "", s)
    s = s.replace("`", "")
    s = re.sub(r'^\s*sql\s*[:\-]*\s*', '', s, flags=re.IGNORECASE)
    s = s.strip().strip('\'"')
    s = re.sub(r'^\s*\n+', '', s)
    s = re.sub(r'\n+\s*$', '', s)

    match = re.search(r'(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE)\b', s, flags=re.IGNORECASE)
    if match:
        s = s[match.start():]

    return s.strip()

# ----------------------------------------------------------------------------
# Suggested queries
# ----------------------------------------------------------------------------
def build_schema_text_from_df(df: pd.DataFrame) -> str:
    parts = []
    for c in df.columns:
        dtype = df[c].dtype
        if pd.api.types.is_integer_dtype(dtype):
            t = "integer"
        elif pd.api.types.is_float_dtype(dtype):
            t = "float"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            t = "datetime"
        elif pd.api.types.is_bool_dtype(dtype):
            t = "boolean"
        else:
            t = "string"
        parts.append(f"{c} ({t})")
    return ", ".join(parts)


def generate_suggested_queries(mdl_text: str, df: pd.DataFrame, max_suggestions: int = 5) -> List[str]:
    """Generate UI-ready queries from schema using LLM."""
    llm = get_llm_provider()
    schema_text = build_schema_text_from_df(df)

    prompt = f"""
You are an expert data analyst. Given the dataset schema below, generate {max_suggestions} short queries.

Schema:
{schema_text}
"""

    try:
        raw = llm.generate(prompt, max_tokens=300, temperature=0.15)
        logger.debug(f"LLM raw suggested-queries output (trimmed 1000): {str(raw)[:1000]}")

        try:
            suggestions = json.loads(raw)
            if isinstance(suggestions, list) and all(isinstance(x, str) for x in suggestions):
                return suggestions[:max_suggestions]
        except Exception:
            logger.debug("Could not parse LLM output as JSON; attempting to clean.")

        lines = [l.strip() for l in re.split(r'[\r\n]+', raw) if l.strip()]
        candidates = []
        cols_lower = {c.lower() for c in df.columns}
        for line in lines:
            line_clean = re.sub(r'^[\-\*\d\.\)\s]+', '', line).strip()
            if '?' in line_clean or any(col in line_clean.lower() for col in cols_lower):
                candidates.append(line_clean)
            if len(candidates) >= max_suggestions:
                break

        if not candidates:
            sentences = re.split(r'(?<=[\.\?])\s+', raw)
            for s in sentences:
                s_clean = s.strip()
                if any(col in s_clean.lower() for col in cols_lower) and len(s_clean) > 10:
                    candidates.append(s_clean)
                if len(candidates) >= max_suggestions:
                    break

        if not candidates:
            cols = list(df.columns)[:6]
            auto = []
            if len(cols) >= 2:
                auto.append(f"What is the distribution of {cols[0]}?")
                auto.append(f"Show average {cols[1]} by {cols[0]}.")
            for c in cols[2:]:
                if len(auto) >= max_suggestions:
                    break
                auto.append(f"Show top 5 {c}.")
            candidates = auto

        return [c.strip() for c in candidates[:max_suggestions]]

    except Exception as e:
        logger.exception(f"Failed to generate suggested queries: {e}")
        return []

# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.post("/api/upload_file", summary="Upload CSV/XLSX and prepare dataset (returns suggested queries)")
async def upload_file(file: UploadFile = File(...), dataset_name: str = "uploaded_dataset", sender: str = "ui_user"):
    request_id = uuid.uuid4().hex[:8]
    start_all = time.time()
    logger.info(f"[req={request_id}] POST /api/upload_file dataset={dataset_name} sender={sender}")

    try:
        save_start = time.time()
        file_path = UPLOAD_DIR / file.filename
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        save_ms = int((time.time() - save_start) * 1000)
        logger.debug(f"[req={request_id}] saved upload {file_path} size={len(content)} bytes time_ms={save_ms}")

        load_start = time.time()
        df = load_dataframe_from_file(file_path)
        DATASETS[dataset_name] = df
        load_ms = int((time.time() - load_start) * 1000)
        logger.info(f"[req={request_id}] loaded dataframe rows={len(df)} cols={len(df.columns)} time_ms={load_ms}")

        mdl_start = time.time()
        mdl = generate_mdl(df, dataset_name)
        mdl_text = mdl_to_text(mdl)
        metadata = {
            "dataset_name": dataset_name,
            "sender": sender,
            "created_at": datetime.utcnow().isoformat(),
            "source_file": str(file_path),
        }
        doc = Document(page_content=mdl_text, metadata=metadata)
        VECTOR_STORE.add_documents([doc], model=EMBEDDING_MODEL_INSTANCE)
        mdl_ms = int((time.time() - mdl_start) * 1000)
        logger.info(f"[req={request_id}] generated MDL and stored embedding dataset={dataset_name} time_ms={mdl_ms}")

        suggest_start = time.time()
        suggested_queries = generate_suggested_queries(mdl_text, df, max_suggestions=5)
        suggest_ms = int((time.time() - suggest_start) * 1000)
        logger.info(f"[req={request_id}] generated suggested_queries count={len(suggested_queries)} time_ms={suggest_ms}")

        total_ms = int((time.time() - start_all) * 1000)
        logger.info(f"[req={request_id}] upload_file completed dataset={dataset_name} total_ms={total_ms}")
        return {
            "status": "success",
            "dataset": dataset_name,
            "message": f"File '{file.filename}' uploaded and processed.",
            "suggested_queries": suggested_queries,
        }

    except Exception as e:
        logger.exception(f"[req={request_id}] Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/query", response_model=QueryResponse, summary="Answer query with summary + chart")
async def process_query(query_input: QueryInput):
    request_id = uuid.uuid4().hex[:8]
    total_start = time.time()
    logger.info(f"[req={request_id}] POST /api/query dataset={query_input.dataset_name} sender={query_input.sender} q=\"{query_input.message}\"")

    # Default response parts
    summary = None
    chart_image = None
    plotly_json = None
    error = None

    try:
        dataset_name = query_input.dataset_name

        # Try get dataframe (may be None)
        df = DATASETS.get(dataset_name)
        mdl = None
        columns = None
        mdl_text_from_store = None

        # Prefer stored MDL text for context when available
        matching_docs = [doc for doc in VECTOR_STORE.documents if doc.metadata.get("dataset_name") == dataset_name]
        if matching_docs:
            try:
                mdl_text_from_store = matching_docs[-1].page_content
            except Exception:
                mdl_text_from_store = None

        if df is not None:
            columns = list(df.columns)
            # Try to construct a minimal DatasetMDL instance (for LLM prompts / clean_sql)
            try:
                mdl_dict = {"dataset": dataset_name, "fields": [{"name": c, "type": "string"} for c in df.columns]}
                mdl = DatasetMDL(**mdl_dict)
            except Exception:
                # defensive minimal object if Pydantic validation fails
                class _MiniMDL:
                    def __init__(self, cols, ds):
                        self.fields = [type("F", (), {"name": c})() for c in cols]
                        self.dataset = ds
                mdl = _MiniMDL(columns, dataset_name)

        # 1) Classify the message (uses local heuristics first, then LLM)
        try:
            label, confidence, reason = classify_user_message(query_input.message, mdl=mdl, columns=columns)
        except Exception as e:
            logger.exception(f"[req={request_id}] classify_user_message failed: {e}")
            label, confidence, reason = ("OUT_OF_SCOPE", 0.25, "classifier failure fallback")

        logger.info(f"[req={request_id}] classifier -> label={label} confidence={confidence:.2f} reason=\"{reason}\"")

        # 2) Handle conversation / out-of-scope immediately with bounded canned replies
        if label == "CONVERSATION":
            # Keep canned replies minimal and safe
            summary = "Hello! I’m your data agent. Upload a dataset (CSV/XLSX) to begin analysis or ask a dataset-specific question when a dataset is loaded."
            return QueryResponse(query=query_input.message, summary=summary, chart_image=None, plotly_json=None, error=None)

        if label == "OUT_OF_SCOPE":
            summary = "Unable to answer, kindly retry with a relevant query."
            return QueryResponse(query=query_input.message, summary=summary, chart_image=None, plotly_json=None, error=None)

        # 3) DATA_QUERY path — dataset must exist
        if df is None:
            error = f"Dataset '{dataset_name}' not found."
            logger.warning(f"[req={request_id}] {error}")
            return QueryResponse(query=query_input.message, summary=None, chart_image=None, plotly_json=None, error=error)

        # 4) Ensure we have an MDL object (prefer stored MDL when available)
        if mdl_text_from_store:
            # ensure mdl has required properties (fields, dataset)
            if not hasattr(mdl, "fields") or not getattr(mdl, "fields"):
                try:
                    mdl = DatasetMDL(dataset=dataset_name, fields=[{"name": c, "type": "string"} for c in df.columns])
                except Exception:
                    pass

        if mdl is None:
            # final fallback: construct minimal DatasetMDL
            mdl_dict = {"dataset": dataset_name, "fields": [{"name": c, "type": "string"} for c in df.columns]}
            mdl = DatasetMDL(**mdl_dict)

        # 5) Generate SQL using LLM (schema-aware)
        sql_start = time.time()
        try:
            sql_raw = generate_sql_query(query_input.message, mdl, dataset_name=dataset_name)
        except Exception as e:
            logger.exception(f"[req={request_id}] SQL generation threw: {e}")
            error = "Could not generate valid SQL."
            return QueryResponse(query=query_input.message, summary=None, chart_image=None, plotly_json=None, error=error)

        sql_gen_ms = int((time.time() - sql_start) * 1000)
        logger.debug(f"[req={request_id}] raw SQL from LLM (trimmed 1000): {str(sql_raw)[:1000]}")

        # 6) Sanitize and clean SQL
        try:
            sanitized_sql = sanitize_sql_from_llm(str(sql_raw))
            logger.debug(f"[req={request_id}] sanitized SQL (trimmed 1000): {sanitized_sql[:1000]}")
        except Exception as e:
            logger.exception(f"[req={request_id}] failed to sanitize LLM SQL: {e}")
            sanitized_sql = str(sql_raw or "")

        if not sanitized_sql or sanitized_sql.strip().upper() == "INVALID QUERY":
            logger.warning(f"[req={request_id}] invalid SQL generated after sanitization")
            error = "Could not generate valid SQL."
            return QueryResponse(query=query_input.message, summary=None, chart_image=None, plotly_json=None, error=error)

        try:
            sql_query = clean_sql(sanitized_sql, mdl)
        except Exception:
            # fallback: ensure it's a string and append semicolon
            sql_query = sanitized_sql.strip().rstrip(";") + ";"

        logger.debug(f"[req={request_id}] cleaned SQL (debug only): \"{sql_query}\"")

        # 7) Execute SQL using duckdb against registered DataFrame
        import duckdb
        try:
            exec_start = time.time()
            con = duckdb.connect()
            con.register("df", df)
            exec_sql = sql_query.replace(f'"{mdl.dataset}"', "df").replace(mdl.dataset, "df")
            result_df = con.execute(exec_sql).fetchdf()
            exec_ms = int((time.time() - exec_start) * 1000)
            logger.info(f"[req={request_id}] executed SQL rows={len(result_df)} time_ms={exec_ms}")
        except Exception as e:
            logger.exception(f"[req={request_id}] SQL execution failed: {e}")
            error = f"SQL execution failed: {e}"
            return QueryResponse(query=query_input.message, summary=None, chart_image=None, plotly_json=None, error=error)

        # 8) Chart generation (df_to_chart_base64 may return tuple (img, json) or single)
        chart_start = time.time()
        try:
            chart_res = df_to_chart_base64(result_df)
            if isinstance(chart_res, tuple):
                chart_image, plotly_json = chart_res
            else:
                chart_image = chart_res
                plotly_json = None
        except Exception as e:
            logger.exception(f"[req={request_id}] chart generation failed: {e}")
            chart_image = None
            plotly_json = None
        chart_ms = int((time.time() - chart_start) * 1000)
        logger.info(f"[req={request_id}] chart generation time_ms={chart_ms} chart_present={chart_image is not None} plotly_json_present={plotly_json is not None}")

        # 9) Summary (LLM)
        summary_start = time.time()
        try:
            summary = summarize_result(result_df, query_input.message)
        except Exception as e:
            logger.exception(f"[req={request_id}] summary generation failed: {e}")
            summary = None
        summary_ms = int((time.time() - summary_start) * 1000)
        logger.info(f"[req={request_id}] summary generation time_ms={summary_ms} summary_len={len(summary or '')}")

        total_ms = int((time.time() - total_start) * 1000)
        logger.info(f"[req={request_id}] query completed total_ms={total_ms}")

        return QueryResponse(query=query_input.message, summary=summary, chart_image=chart_image, plotly_json=plotly_json, error=None)

    except Exception as e:
        logger.exception(f"[req={request_id}] unexpected failure: {e}")
        return QueryResponse(query=query_input.message, summary=None, chart_image=None, plotly_json=None, error=f"Query failed: {str(e)}")


@app.get("/health", include_in_schema=False)
async def health_check():
    return {
        "status": "healthy",
        "datasets_loaded": list(DATASETS.keys()),
        "vector_store_docs": len(VECTOR_STORE.documents),
        "llm_metrics": {
            "llm_success": getattr(__import__('mdl_utils'), 'LLM_SUCCESS_COUNT', 0),
            "llm_parse_fail": getattr(__import__('mdl_utils'), 'LLM_PARSE_FAIL_COUNT', 0),
            "llm_error": getattr(__import__('mdl_utils'), 'LLM_ERROR_COUNT', 0),
        },
    }


@app.get("/api/get_schema", summary="Get stored MDL text + suggested queries")
async def get_schema(dataset_name: str):
    try:
        matching_docs = [doc for doc in VECTOR_STORE.documents if doc.metadata.get("dataset_name") == dataset_name]
        if not matching_docs:
            raise HTTPException(status_code=404, detail=f"No MDL found for {dataset_name}")
        mdl_text = matching_docs[-1].page_content
        df = DATASETS.get(dataset_name)
        suggested = []
        if df is not None:
            suggested = generate_suggested_queries(mdl_text, df, max_suggestions=5)
        return {"dataset": dataset_name, "mdl_text": mdl_text, "suggested_queries": suggested}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("get_schema failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
