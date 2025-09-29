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
from typing import Dict, Any, Optional

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
)

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")]
)
logger = logging.getLogger(__name__)

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
    query: str
    # sql_query intentionally not returned to client
    summary: Optional[str] = None
    chart_image: Optional[str] = None
    error: Optional[str] = None

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def df_to_chart_base64(df: pd.DataFrame) -> Optional[str]:
    """Generate a simple chart from first 2 columns and return as base64 image (Plotly)."""
    try:
        if df.shape[0] == 0:
            logger.debug("df_to_chart_base64: empty dataframe, skipping chart")
            return None

        if df.shape[1] >= 2:
            x_col, y_col = df.columns[:2]
            fig = px.bar(df, x=x_col, y=y_col, title="Auto-generated chart")
        else:
            fig = px.bar(df)

        # Export figure as PNG (bytes) - requires kaleido
        img_bytes = fig.to_image(format="png")
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"

    except Exception as e:
        logger.exception(f"Plotly chart generation failed: {e}")
        return None


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

# ----------------------------------------------------------------------------
# LLM SQL sanitizer
# ----------------------------------------------------------------------------
def sanitize_sql_from_llm(raw: str) -> str:
    """
    Remove markdown fences, backticks, tildes, leading 'sql' labels, and stray quotes.
    Keep the main SQL statement text.
    """
    if not raw:
        return raw

    s = raw

    # Remove common markdown code fences ```sql ... ```
    s = re.sub(r"```(?:\s*sql)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```", "", s)

    # Remove triple tildes fences
    s = re.sub(r"~~~(?:\s*sql)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*~~~", "", s)

    # Remove inline backticks
    s = s.replace("`", "")

    # Remove leading language labels like "sql\n" or "SQL:" etc.
    s = re.sub(r'^\s*sql\s*[:\-]*\s*', '', s, flags=re.IGNORECASE)

    # Strip leading/trailing whitespace and quotes
    s = s.strip().strip('\'"')

    # Collapse multiple newlines at start/end
    s = re.sub(r'^\s*\n+', '', s)
    s = re.sub(r'\n+\s*$', '', s)

    # If the LLM returned extraneous surrounding text, attempt to extract the first SQL-looking block:
    # naive extraction: look for the first 'SELECT' or common SQL keywords and cut starting there
    match = re.search(r'(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE)\b', s, flags=re.IGNORECASE)
    if match:
        s = s[match.start():]

    return s.strip()

# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.post("/api/upload_file", summary="Upload CSV/XLSX and prepare dataset")
async def upload_file(file: UploadFile = File(...), dataset_name: str = "uploaded_dataset", sender: str = "ui_user"):
    request_id = uuid.uuid4().hex[:8]
    start_all = time.time()
    logger.info(f"[req={request_id}] POST /api/upload_file dataset={dataset_name} sender={sender}")

    try:
        # Save uploaded file
        save_start = time.time()
        file_path = UPLOAD_DIR / file.filename
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        save_ms = int((time.time() - save_start) * 1000)
        logger.debug(f"[req={request_id}] saved upload {file_path} size={len(content)} bytes time_ms={save_ms}")

        # Load dataframe
        load_start = time.time()
        df = load_dataframe_from_file(file_path)
        DATASETS[dataset_name] = df
        load_ms = int((time.time() - load_start) * 1000)
        logger.info(f"[req={request_id}] loaded dataframe rows={len(df)} cols={len(df.columns)} time_ms={load_ms}")

        # Generate MDL + store in vector DB
        mdl_start = time.time()
        mdl = generate_mdl(df, dataset_name)
        mdl_text = mdl_to_text(mdl)
        metadata = {
            "dataset_name": dataset_name,
            "sender": sender,
            "created_at": datetime.utcnow().isoformat(),
            "source_file": str(file_path)
        }
        doc = Document(page_content=mdl_text, metadata=metadata)
        VECTOR_STORE.add_documents([doc], model=EMBEDDING_MODEL_INSTANCE)
        mdl_ms = int((time.time() - mdl_start) * 1000)
        logger.info(f"[req={request_id}] generated MDL and stored embedding dataset={dataset_name} time_ms={mdl_ms}")

        total_ms = int((time.time() - start_all) * 1000)
        logger.info(f"[req={request_id}] upload_file completed dataset={dataset_name} total_ms={total_ms}")

        return {"status": "success", "dataset": dataset_name, "message": f"File '{file.filename}' uploaded and processed."}

    except Exception as e:
        logger.exception(f"[req={request_id}] Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/query", response_model=QueryResponse, summary="Answer query with summary + chart")
async def process_query(query_input: QueryInput):
    request_id = uuid.uuid4().hex[:8]
    total_start = time.time()
    logger.info(f"[req={request_id}] POST /api/query dataset={query_input.dataset_name} sender={query_input.sender} q=\"{query_input.message}\"")

    try:
        dataset_name = query_input.dataset_name
        if dataset_name not in DATASETS:
            logger.warning(f"[req={request_id}] dataset not found: {dataset_name}")
            return QueryResponse(query=query_input.message, error=f"Dataset '{dataset_name}' not found.")

        df = DATASETS[dataset_name]

        # Get MDL from vector store (used only to provide schema context to LLM)
        matching_docs = [doc for doc in VECTOR_STORE.documents if doc.metadata.get("dataset_name") == dataset_name]
        if not matching_docs:
            logger.warning(f"[req={request_id}] no MDL document found for dataset {dataset_name}")
            return QueryResponse(query=query_input.message, error=f"No MDL found for dataset '{dataset_name}'.")

        # For auditing, log the stored MDL text at DEBUG level (trimmed)
        mdl_text = matching_docs[-1].page_content
        logger.debug(f"[req={request_id}] retrieved MDL (trimmed): {mdl_text[:500].replace(chr(10),' ')}")

        # Construct a simple DatasetMDL placeholder for SQL generation using schema names
        # (generate_sql_query expects a DatasetMDL; keeping it minimal for demo)
        mdl_dict = {"dataset": dataset_name, "fields": [{"name": c, "type": "string"} for c in df.columns]}
        mdl = DatasetMDL(**mdl_dict)

        # Generate SQL
        sql_start = time.time()
        sql_query = generate_sql_query(query_input.message, mdl, dataset_name=dataset_name)
        sql_gen_ms = int((time.time() - sql_start) * 1000)

        # Log raw LLM output for debugging
        logger.debug(f"[req={request_id}] raw SQL from LLM (trimmed 1000): {str(sql_query)[:1000]}")

        # Sanitize LLM output to remove markdown/code fences and stray punctuation
        try:
            sanitized_sql = sanitize_sql_from_llm(str(sql_query))
            logger.debug(f"[req={request_id}] sanitized SQL (trimmed 1000): {sanitized_sql[:1000]}")
        except Exception as e:
            logger.exception(f"[req={request_id}] failed to sanitize LLM SQL: {e}")
            sanitized_sql = sql_query or ""

        if not sanitized_sql or sanitized_sql.strip().upper() == "INVALID QUERY":
            logger.warning(f"[req={request_id}] invalid SQL generated after sanitization")
            return QueryResponse(query=query_input.message, error="Could not generate valid SQL.")

        # Clean SQL (adds semicolon / handles fallback)
        sql_query = clean_sql(sanitized_sql, mdl)
        logger.debug(f"[req={request_id}] cleaned SQL (debug only): \"{sql_query}\"")

        # Execute SQL (use duckdb registered dataframe)
        import duckdb
        try:
            exec_start = time.time()
            con = duckdb.connect()
            con.register("df", df)
            # Replace the dataset name in the SQL with the registered table 'df'
            exec_sql = sql_query.replace(f'"{mdl.dataset}"', "df").replace(mdl.dataset, "df")
            result_df = con.execute(exec_sql).fetchdf()
            exec_ms = int((time.time() - exec_start) * 1000)
            logger.info(f"[req={request_id}] executed SQL rows={len(result_df)} time_ms={exec_ms}")
        except Exception as e:
            logger.exception(f"[req={request_id}] SQL execution failed: {e}")
            return QueryResponse(query=query_input.message, error=f"SQL execution failed: {e}")

        # Generate chart + summary
        chart_start = time.time()
        chart_image = df_to_chart_base64(result_df)
        chart_ms = int((time.time() - chart_start) * 1000)
        logger.info(f"[req={request_id}] chart generation time_ms={chart_ms} chart_present={chart_image is not None}")

        summary_start = time.time()
        summary = summarize_result(result_df, query_input.message)
        summary_ms = int((time.time() - summary_start) * 1000)
        logger.info(f"[req={request_id}] summary generation time_ms={summary_ms} summary_len={len(summary or '')}")

        total_ms = int((time.time() - total_start) * 1000)
        logger.info(f"[req={request_id}] query completed total_ms={total_ms}")

        return QueryResponse(
            query=query_input.message,
            summary=summary,
            chart_image=chart_image
        )

    except Exception as e:
        logger.exception(f"[req={request_id}] Query failed: {e}")
        return QueryResponse(query=query_input.message, error=f"Query failed: {str(e)}")


@app.get("/health", include_in_schema=False)
async def health_check():
    return {
        "status": "healthy",
        "datasets_loaded": list(DATASETS.keys()),
        "vector_store_docs": len(VECTOR_STORE.documents)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
