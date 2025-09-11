#!/usr/bin/env python3

from __future__ import annotations

import os
import json
import logging
import sys
import tempfile
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import numpy as np

# DeepSeek API configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_AVAILABLE = True  # We'll handle the actual availability in the function
st.session_state.setdefault("uploaded_file", None)
st.session_state.setdefault("reset_local_dataset_select", False)
# Load environment variables from .env file
load_dotenv()

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Debug: Check if DEEPSEEK_API_KEY is loaded
if not os.getenv("DEEPSEEK_API_KEY"):
    logger.warning("DEEPSEEK_API_KEY not found in environment variables")
    logger.info("Current working directory: %s", os.getcwd())
    logger.info("Environment variables: %s", {k: v for k, v in os.environ.items() if "DEEPSEEK" in k.upper()})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Local utilities
from db_utils import (
    VectorStore,
    create_embeddings,
)

# MDL utilities (from your provided module)
from mdl_utils import generate_mdl, mdl_to_text, generate_sql_query

# ---------------------------
# Logging & Environment
# ---------------------------

# Load environment variables (supports .env)
load_dotenv()

# ---------------------------
# Utilities for safe arithmetic evaluation in JSON
# ---------------------------


def safe_eval_arithmetic(expr: Any) -> Any:
    """Safely evaluate simple arithmetic expressions used in chart configs."""
    if not isinstance(expr, str):
        return expr

    # Skip color codes or non-numeric strings
    if any(token in expr for token in ("rgba", "#")):
        return expr

    allowed = set("0123456789+*/-(). ")
    if not set(expr) <= allowed:
        return expr

    try:
        # Extremely limited eval: no builtins
        return eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return expr

def process_chart_config(config: Any) -> Any:
    """Recursively process chart config to evaluate arithmetic expressions."""
    if isinstance(config, dict):
        return {k: process_chart_config(v) for k, v in config.items()}
    if isinstance(config, list):
        return [process_chart_config(v) for v in config]
    if isinstance(config, str):
        if any(op in config for op in "+-*/"):
            evaluated = safe_eval_arithmetic(config)
            return evaluated
    return config


def evaluate_arithmetic_in_json(json_str: str) -> str:
    """Evaluate arithmetic expressions inside JSON arrays/values in a string."""
    import re

    def repl_array(match: "re.Match[str]") -> str:
        content = match.group(0)

        def eval_match(m: "re.Match[str]") -> str:
            candidate = m.group(0)
            out = safe_eval_arithmetic(candidate)
            return str(out) if isinstance(out, (int, float)) else candidate

        return re.sub(r"(?<![\w.])(?:[\d.]+\s*[+\-*/]\s*)+[\d.()]+", eval_match, content)

    array_pat = r"\[[^\[\]]*\]"
    prev = None
    curr = json_str
    while prev != curr:
        prev = curr
        curr = __import__("re").sub(array_pat, repl_array, curr, flags=__import__("re").DOTALL)
    return curr


# ---------------------------
# Lazy / cached resources
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_vector_store() -> VectorStore:
    # 384 dims for all-MiniLM-L6-v2 (if used)
    return VectorStore(dimension=384)


@st.cache_resource(show_spinner=False)
def load_sentence_transformer():
    """Load SentenceTransformer lazily and safely.
    Returns the model object with .encode method.
    """
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SentenceTransformer model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Verify the model has the encode method
        if not hasattr(model, "encode") or not callable(model.encode):
            raise RuntimeError("Loaded model does not have a callable 'encode' method")
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer: {str(e)}", exc_info=True)
        raise RuntimeError(
            "Failed to load SentenceTransformer. Please ensure it's installed: "
            "pip install -U sentence-transformers"
        ) from e
# ---------------------------
# App Title
# ---------------------------
st.title("Agent Graphiq: AI Chart Generator")

# Vector store (cached)
vector_store = get_vector_store()


# ---------------------------
# File upload & parsing + MDL creation
# ---------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload your data file (CSV or Excel)", type=["csv", "xlsx", "xls"]
)


def get_file_hash(uploaded) -> str:
    import hashlib

    return f"{hashlib.md5(uploaded.getvalue()).hexdigest()}_{uploaded.name}"


def extract_features(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "columns": list(df.columns),
        "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=["datetime"]).columns.tolist(),
    }


def ensure_collection_for_file(uploaded, vs: VectorStore) -> Optional[str]:
    """Process the uploaded file and create a new collection with embeddings.
    
    This function always processes the file as a new dataset, regardless of whether
    embeddings for this file exist in the vector store.
    """
    if not uploaded:
        st.sidebar.warning("⚠️ No file provided. Please upload a file to create embeddings.")
        return None

    try:
        # Always create a new collection with fresh embeddings
        return create_new_collection(uploaded, vs)

    except Exception as e:
        logger.exception("Error processing file: %s", e)
        st.sidebar.error(f"Error processing file: {e}")
        return None


def coerce_numeric_safe(series: pd.Series, threshold: float = 0.8) -> pd.Series:
    """
    Try to convert a column to numeric.
    If too many values fail conversion, keep it as object.
    
    Args:
        series: Pandas Series
        threshold: fraction of values that must be convertible to numeric
    """
    if series.empty:
        return series

    # Attempt conversion
    coerced = pd.to_numeric(series, errors="coerce")
    non_null_fraction = coerced.notnull().mean()

    # If most values are numeric, keep coerced (non-numeric become NaN/NULL)
    if non_null_fraction >= threshold:
        return coerced
    else:
        return series  # keep as is (likely categorical)


def create_new_collection(uploaded, vs: VectorStore) -> Optional[str]:
    """Process uploaded file and create MDL embedding."""
    if not uploaded:
        st.error("No file provided for creating embeddings.")
        return None

    logger.info("Processing uploaded file for MDL generation…")

    tmp_path = os.path.join(tempfile.gettempdir(), uploaded.name)
    try:
        # write a temporary copy (Streamlit file uploader is in-memory)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        try:
            if uploaded.name.lower().endswith(".csv"):
                # Try reading with default settings first
                try:
                    df = pd.read_csv(tmp_path)
                except pd.errors.ParserError as e:
                    # If parsing fails, try with error_bad_lines=False and warn_bad_lines=True
                    logger.warning(f"CSV parsing error, trying with more permissive settings: {e}")
                    df = pd.read_csv(tmp_path, on_bad_lines='warn')
            else:
                df = pd.read_excel(tmp_path)

            df = df.dropna(how="all")
            if df.empty:
                st.error("The uploaded file is empty or contains no valid data.")
                return None
                
        except Exception as e:
            logger.exception(f"Error reading file: {e}")
            st.error(f"Error reading file: {e}\nPlease check the file format and try again.")
            return None

        # save df to session
        st.session_state.df = df
        st.session_state.last_uploaded_file = uploaded.name
        file_hash = get_file_hash(uploaded)
        
        # Load encoder model
        model = load_sentence_transformer()

        # Generate and embed MDL only
        try:
            mdl = generate_mdl(df, dataset_name=file_hash)
            mdl_text = mdl_to_text(mdl)
            
            # Store MDL in session state
            st.session_state.mdl = mdl
            st.session_state.mdl_text = mdl_text

            # Create embedding for the MDL only
            create_embeddings(
                texts=[mdl_text],
                metadatas=[{'type': 'mdl', 'file_hash': file_hash}],
                ids=[f"{file_hash}_mdl"],
                file_hash=file_hash,
                model=model,
                vector_store=vs,
            )
            
            st.sidebar.success("✅ Successfully created MDL embedding")
            return file_hash
        except Exception as e:
            logger.exception("MDL generation/embedding failed: %s", e)
            st.sidebar.warning("⚠️ MDL generation failed; continuing without MDL.")

        return file_hash

    except Exception as e:
        logger.exception("Error creating embeddings: %s", e)
        st.sidebar.error(f"Error creating embeddings: {e}")
        return None
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

# ---------- Local dataset selector (loads files from ./Dataset) ----------
from types import SimpleNamespace
 
def list_local_datasets(folder: str = "Dataset", max_files: int = 10):
    """
    Returns up to `max_files` dataset filenames in the given folder.
    Only CSV/XLSX/XLS are returned and sorted for stable order.
    """
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return []
    candidates = sorted([f for f in p.iterdir() if f.suffix.lower() in (".csv", ".xlsx", ".xls")])
    return [str(f.name) for f in candidates[:max_files]]
 
class LocalUploadedFile:
    """
    Minimal wrapper to mimic Streamlit's UploadedFile interface enough for
    create_new_collection/get_file_hash usage (getbuffer and getvalue and name).
    """
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.name = self.path.name
        # read bytes lazily
        self._bytes = None
 
    def _ensure(self):
        if self._bytes is None:
            with open(self.path, "rb") as f:
                self._bytes = f.read()
 
    def getbuffer(self):
        # create a buffer-like object that has a .tobytes() if needed by code
        self._ensure()
        return memoryview(self._bytes)
 
    def getvalue(self):
        self._ensure()
        return self._bytes
 
# Show the selector only when the Dataset folder exists
local_files = list_local_datasets(folder="Dataset", max_files=5)
dataset_choice = None
 
if local_files:
    options = ["(Select)"] + local_files
 
    # If an upload happened earlier, a reset flag will be present. Consume it here.
    reset = st.session_state.pop("reset_local_dataset_select", False)
 
    # Determine default index for the selectbox:
    # - if reset flag present -> show "(Select)" (index 0)
    # - otherwise, keep previous selection if valid, else default to 0
    if reset:
        default_index = 0
    else:
        prev = st.session_state.get("local_dataset_select", "(Select)")
        default_index = options.index(prev) if prev in options else 0
 
    dataset_choice = st.sidebar.selectbox(
        "Or choose a sample dataset from project (Dataset/)",
        options,
        index=default_index,
        key="local_dataset_select",
        help="Pick one of the datasets from the project's Dataset/ folder to load it directly."
    )
 
# If the user picked a local file, create a LocalUploadedFile and process it
if dataset_choice and dataset_choice != "(Select)":
    local_path = Path("Dataset") / dataset_choice
    if local_path.exists():
        # wrap it so existing create_new_collection works unchanged
        pseudo_uploaded = LocalUploadedFile(local_path)
        st.sidebar.info(f"Loading local dataset: {dataset_choice}")
        file_hash = ensure_collection_for_file(pseudo_uploaded, vector_store)
        if file_hash:
            # Cancel any uploaded file when a dataset is selected
            st.session_state.last_uploaded_file = None
            st.session_state.df = st.session_state.df  # keep current DF
            st.session_state.uploaded_file = None
            st.session_state.current_file_hash = file_hash
            st.success(f"Loaded dataset: {dataset_choice}")
            # optionally scroll to main area to show data preview (no structural change)
            # Note: we don't call st.experimental_rerun() so that the flow continues naturally.
    else:
        st.sidebar.error(f"Dataset file missing: {local_path}")
       
# -------------------------------------------------------------------------------------------------------
 
# ---------------------------
# Load / show data
# ---------------------------
if uploaded_file is not None:
    file_hash = ensure_collection_for_file(uploaded_file, vector_store)
    if file_hash:
        st.session_state["reset_local_dataset_select"] = True
        st.session_state.current_file_hash = file_hash
 

st.session_state.setdefault("df", None)
st.session_state.setdefault("current_file_hash", None)
st.session_state.setdefault("last_uploaded_file", None)
st.session_state.setdefault("mdl", None)
st.session_state.setdefault("mdl_text", None)

if st.session_state.df is not None and not st.session_state.df.empty:
    df = st.session_state.df
    st.sidebar.info(f"📊 Using embeddings for: {st.session_state.last_uploaded_file or 'current file'}")
    st.subheader("Data Preview (First 5 Rows)")
    st.dataframe(df.head(5))
    # with st.sidebar.expander("Dataset MDL (auto-generated)"):
    #     if st.session_state.mdl_text:
    #         st.text_area("MDL (brief)", st.session_state.mdl_text, height=240)
    #     else:
    #         st.write("MDL not available for this dataset.")
else:
    st.info("Please upload a data file to get started.")
    st.stop()


# ---------------------------
# Query & Chart Generation (MDL-aware)
# ---------------------------
user_prompt = st.text_input("Enter your query or request for analysis:")


def clean_sql_query(sql: str) -> str:
    """Clean SQL query by removing markdown code block syntax if present."""
    # Remove markdown code block syntax if present
    if sql.strip().startswith("```sql"):
        lines = sql.split('\n')
        # Remove first and last line (```sql and ```)
        sql = '\n'.join(lines[1:-1])
    return sql.strip()

def is_numeric(value: Any) -> bool:
    """Check if a value can be converted to a number."""
    try:
        float(str(value).strip())
        return True
    except (ValueError, TypeError):
        return False

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string/object columns to numeric where safe, replacing bad values with NULL."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = coerce_numeric_safe(df[col])
    return df

def normalize_chart_config(chart_config):
    """Normalize LLM chart config to expected format."""
    # Handle the case where chart_config is a list of charts
    if isinstance(chart_config, list):
        return [normalize_chart_config(config) for config in chart_config]
        
    # Handle the new LLM response format (with x, y, labels, values)
    if 'x' in chart_config and 'y' in chart_config:
        # This is a bar/line chart with x and y values
        return {
            "data": {
                "labels": chart_config['x'],
                "datasets": [{
                    "data": chart_config['y'],
                    "backgroundColor": chart_config.get('color', None)
                }]
            },
            "options": {
                "title": {"text": chart_config.get("title", "")},
                "scales": {
                    "x": {"title": {"text": chart_config.get("x_label", "")}},
                    "y": {"title": {"text": chart_config.get("y_label", "")}}
                }
            },
            "type": chart_config.get("type", "bar"),
            "orientation": chart_config.get("orientation", "v")
        }
    elif 'values' in chart_config and 'labels' in chart_config:
        # This is a pie/donut chart with values and labels
        return {
            "data": {
                "labels": chart_config['labels'],
                "datasets": [{
                    "data": chart_config['values'],
                    "backgroundColor": chart_config.get('color', None)
                }]
            },
            "options": {
                "title": {"text": chart_config.get("title", "")}
            },
            "type": chart_config.get("type", "pie")
        }
    return chart_config

def render_chart(chart_config):
    """Render a chart based on the provided configuration."""
    import streamlit as st
    import json
    import plotly.express as px
    import pandas as pd
    import numpy as np
    
    try:
        # If we have a list of charts, render each one
        if isinstance(chart_config, list):
            for chart in chart_config:
                render_chart(chart)
            return
            
        # Normalize the chart config
        chart_config = normalize_chart_config(chart_config)
        
        chart_type = chart_config.get('type', 'bar')
        data = chart_config.get('data', {})
        options = chart_config.get('options', {})
        
        # Get the first dataset (supporting multiple datasets in the future)
        dataset = data.get('datasets', [{}])[0] if data.get('datasets') else {}
        
        # Create a dataframe from the chart data
        if 'labels' in data and 'data' in dataset:
            df = pd.DataFrame({
                'labels': data['labels'],
                'values': dataset['data']
            })
        else:
            df = pd.DataFrame()
        
        # Get color sequence and ensure it's a list with valid colors
        color_sequence = None
        if 'backgroundColor' in dataset:
            color_sequence = dataset['backgroundColor']
            if not isinstance(color_sequence, (list, tuple)):
                color_sequence = [color_sequence] if color_sequence else None
        
        # Get the title from options or root level
        title = options.get('title', {}).get('text', '') if isinstance(options, dict) else ''
        if not title and 'title' in chart_config:
            title = chart_config['title']
        
        # Handle different chart types
        if chart_type in ['pie', 'doughnut']:
            if not df.empty:
                fig = px.pie(
                    df, 
                    values='values', 
                    names='labels',
                    title=title,
                    hole=0.4 if chart_type == 'doughnut' else 0,
                    color_discrete_sequence=color_sequence or px.colors.qualitative.Plotly
                )
            
        elif chart_type in ['bar', 'horizontalBar']:
            orientation = 'h' if chart_type == 'horizontalBar' else 'v'
            if not df.empty:
                fig = px.bar(
                    df,
                    x='values' if orientation == 'h' else 'labels',
                    y='labels' if orientation == 'h' else 'values',
                    title=title,
                    orientation=orientation,
                    color_discrete_sequence=color_sequence or [chart_config.get('color')] if 'color' in chart_config else None
                )
                
                # Update axis labels if provided
                if 'scales' in options:
                    if 'x' in options['scales'] and 'title' in options['scales']['x']:
                        fig.update_xaxes(title_text=options['scales']['x']['title'].get('text', ''))
                    if 'y' in options['scales'] and 'title' in options['scales']['y']:
                        fig.update_yaxes(title_text=options['scales']['y']['title'].get('text', ''))
        else:
            st.warning(f"Unsupported chart type: {chart_type}")
            return
        
        # Update layout
        if 'fig' in locals():
            fig.update_layout(
                showlegend=chart_type in ['pie', 'doughnut'],
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                height=400,
                title_x=0.5,
                title_y=0.95
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        import traceback
        error_details = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc(),
            'chart_json': clean_json if 'clean_json' in locals() else 'Not available',
            'chart_configs': chart_configs if 'chart_configs' in locals() else 'Not available',
            'config': config if 'config' in locals() else 'Not available'
        }
        logger.error("Detailed error information: %s", json.dumps(error_details, indent=2, default=str))
        
        # Show a simplified error to the user
        st.error(f"Error generating visualization: {str(e)}")
        
        # Show more details in an expander for debugging
        with st.expander("Click for error details"):
            st.write("### Error Details")
            st.code(traceback.format_exc())
            
            if 'clean_json' in locals():
                st.write("### Chart JSON")
                st.code(clean_json)
            
            if 'chart_configs' in locals():
                st.write("### Parsed Chart Configs")
                st.json(chart_configs)
            
            if 'config' in locals():
                st.write("### Current Chart Config")
                st.json(config)

def execute_sql_on_df(sql: str, df: pd.DataFrame, max_retries: int = 2) -> pd.DataFrame:
    """
    Execute SQL on the provided DataFrame with robust error handling and type validation.
    Uses pandasql for better SQL compatibility.
    """
    import re
    from pandasql import sqldf
    
    # Clean and prepare the SQL query
    sql = clean_sql_query(sql)
    sql = sql.strip().rstrip(';')
    
    # Log the original query for debugging with more visibility
    print("\n" + "="*80)
    print(f"[DEBUG] Original SQL query:\n{sql}")
    print("="*80 + "\n")
    logger.info(f"Original SQL query: {sql}")

    # Common SQL injection prevention
    if any(keyword in sql.upper() for keyword in ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE']):
        raise ValueError("Modification queries are not allowed")
        
    # Check if the query is a SELECT query
    if not sql.strip().upper().startswith('SELECT'):
        raise ValueError("Only SELECT queries are allowed")
    
    # Create a clean copy of the dataframe with lowercase column names
    df_clean = df.copy()
    df_clean.columns = [str(col).lower() for col in df_clean.columns]
    
    # Extract the table name if it exists - handle various quoting styles and special characters
    table_match = re.search(
        r'(?i)(from\s+)([`"\[\]]?[^\s`"\[\];]+)(?=\s|;|$)', 
        sql
    )
    if table_match:
        prefix = table_match.group(1)  # "from "
        original_table = table_match.group(2).strip('`"[]')
        print(f"[DEBUG] Found table reference: '{original_table}'")
        logger.info(f"Found table reference: '{original_table}'")
        
        # Replace ONLY the table reference, not the rest of the query
        sql = sql[:table_match.start(2)] + "df_clean" + sql[table_match.end(2):]
        print(f"[DEBUG] Replaced table name safely. New query:\n{sql}")
    else:
        print("[DEBUG] No table reference found in SQL query")
        if "FROM" not in sql.upper() and "WHERE" in sql.upper():
            where_pos = sql.upper().find("WHERE")
            sql = sql[:where_pos] + "FROM df_clean " + sql[where_pos:]
            print(f"[DEBUG] Added missing FROM clause. New query:\n{sql}")
        elif "FROM" not in sql.upper():
            sql = sql + " FROM df_clean"
            print(f"[DEBUG] Added missing FROM clause at the end. New query:\n{sql}")
    # Convert all column references to lowercase
    print("[DEBUG] Available columns in DataFrame:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col} (type: {type(col)})")
    
    for col in df.columns:
        col_lower = str(col).lower()
        if col != col_lower:
            print(f"[DEBUG] Converting column name to lowercase: '{col}' -> '{col_lower}'")
            # Replace column names in SELECT, WHERE, GROUP BY, ORDER BY, HAVING
            sql_before = sql
            sql = re.sub(
                rf'(?<![a-zA-Z0-9_`".]){re.escape(col)}(?![a-zA-Z0-9_`"])', 
                f'"{col_lower}"', 
                sql, 
                flags=re.IGNORECASE
            )
            if sql_before != sql:
                print(f"[DEBUG] Column name replacement modified the query")
                # print(f"Before: {sql_before}")
                # print(f"After:  {sql}")
                # print("-"*40)
    
    # Handle backtick-quoted column names
    sql_before = sql
    sql = re.sub(r'`([^`]+)`', r'"\1"', sql)
    if sql_before != sql:
        print(f"[DEBUG] Backtick replacement modified the query")

    # --- START FALLBACK: ensure top-level FROM maps to df_clean ---
    # If replacement didn't map the table to df_clean, replace the first FROM <something>
    # with FROM df_clean. This avoids issues where the LLM used a filename/table name.
    if 'df_clean' not in sql.lower():
        # Replace only the first top-level FROM <table> occurrence (case-insensitive)
        sql = re.sub(r'(?i)\bFROM\s+([`"\[]?[^\s;]+[`"\]]?)', 'FROM df_clean', sql, count=1)
        print("[DEBUG] Applied fallback replacement to map FROM -> df_clean")
    # --- END FALLBACK ---

    print("\n" + "="*80)
    print(f"[DEBUG] Final SQL to be executed:\n{sql}")
    print("="*80 + "\n")
    logger.info(f"Modified SQL: {sql}")
    
    try:
        # Execute the query using pandasql with a dictionary of available dataframes
        result = sqldf(sql, {'df_clean': df_clean})
        
        # If the result is empty but the query should return something,
        # try to provide more helpful error information
        if result.empty and ('count(' in sql.lower() or 'sum(' in sql.lower()):
            logger.warning("Empty result for aggregation query")
            # Show sample data for debugging
            sample = df_clean.head(5)
            logger.info(f"Sample data (first 5 rows):\n{sample}")
            
        return result
        
    except Exception as e:
        # Try one more time with the original column names if we were using lowercase
        try:
            logger.warning("Retrying with original column names...")
            return sqldf(sql, {'df_clean': df})
        except Exception as e2:
            # logger.error(f"SQL execution failed: {str(e2)}")
            # Provide more detailed error information
            sample_columns = ", ".join([f'"{col}"' for col in df_clean.columns])
            raise ValueError(
                f"Failed to execute SQL: {str(e2)}\n\n"
                f"Available columns: {sample_columns}\n"
                f"Sample data (first row): {df_clean.iloc[0].to_dict() if not df_clean.empty else 'No data'}"
            )

if st.button("Generate Charts") and user_prompt:
    try:
        model = load_sentence_transformer()
        query_embedding = model.encode(user_prompt, convert_to_numpy=True).tolist()

        # Similarity search across file collection (rows + mdl)
        similar_docs = vector_store.similarity_search(
            query_embedding=query_embedding,
            k=100,
            file_hash=st.session_state.current_file_hash,
        )

        # Determine whether the top result is MDL
        top_is_mdl = False
        top_doc = None
        if similar_docs:
            top_doc = similar_docs[0]
            top_meta = top_doc.get("metadata", {}) or top_doc.get("meta", {}) or {}
            top_is_mdl = top_meta.get("type") == "mdl" or (top_doc.get("id", "").endswith("_mdl"))

        # If top match is MDL, ask LLM to produce SQL, execute it, and use result for chart generation
        query_result_df = None
        generated_sql = None

        if top_is_mdl and st.session_state.mdl is not None:
            # Use MDL-aware SQL generation
            try:
                generated_sql = generate_sql_query(
                    natural_language_query=user_prompt,
                    mdl=st.session_state.mdl,
                    dataset_name=st.session_state.current_file_hash,
                )
                # Check for INVALID QUERY before starting the spinner
                if isinstance(generated_sql, str) and generated_sql.strip().upper() == "INVALID QUERY":
                    st.warning("⚠️ The query appears unrelated to this dataset. Please rephrase or try another dataset.")
                    st.stop()
                    
                with st.spinner("Analyzing your data…"):
                    logger.info(f"Generated SQL: {generated_sql}")
                    st.sidebar.code(generated_sql, language="sql")

                    # Execute SQL on in-memory df
                    query_result_df = execute_sql_on_df(generated_sql, st.session_state.df)

                    if query_result_df is None or query_result_df.empty:
                        st.warning("SQL returned no rows — falling back to full dataset for chart generation.")
                        query_result_df = st.session_state.df
                    
                    # Display the filtered data table if we have results
                    st.subheader("Query Results")
                    st.dataframe(
                        query_result_df.head(100),  # Limit to first 100 rows for display
                        use_container_width=True,
                        height=min(400, 35 * (min(100, len(query_result_df)) + 1)),
                        hide_index=True
                    )
                    st.caption(f"Showing {min(100, len(query_result_df))} of {len(query_result_df)} rows")
                    
                    # Add a download button for the filtered data
                    if not query_result_df.empty:
                        csv = query_result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download filtered data as CSV",
                            data=csv,
                            file_name=f'filtered_data_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                            mime='text/csv',
                        )
                    
                    st.sidebar.success(f"✅ SQL executed: {len(query_result_df)} rows returned")
            except Exception as e:
                logger.exception("SQL generation/execution failed: %s", e)
                st.sidebar.error(f"SQL generation/execution failed: {e}")
                query_result_df = st.session_state.df.head(100)
        else:
            # Fallback: use top similar rows as context (or sample of df)
            if similar_docs:
                # Build a small dataframe from returned docs' metadata if available
                rows = []
                for d in similar_docs:
                    meta = d.get("metadata") or d.get("meta") or {}
                    # we expect metadata to be original row dict
                    if isinstance(meta, dict) and set(meta.keys()) & set(st.session_state.df.columns):
                        rows.append(meta)
                if rows:
                    query_result_df = pd.DataFrame(rows)
                else:
                    query_result_df = st.session_state.df.head(100)
            else:
                query_result_df = st.session_state.df.head(100)

        def get_llm_provider():
            """Initialize and return the LLM provider (DeepSeek)."""
            import requests
            
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable not set")
            
            def generate(prompt: str) -> str:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
                
                try:
                    response = requests.post(
                        DEEPSEEK_API_URL,
                        headers=headers,
                        json=data
                    )
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"]
                except requests.RequestException as e:
                    logger.error(f"Error calling DeepSeek API: {str(e)}")
                    raise RuntimeError(f"Failed to call DeepSeek API: {str(e)}")
            
            class LLMProvider:
                def __init__(self):
                    self.generate = generate
            
            return LLMProvider()

        def generate_insights(prompt_text, sample_data, generated_sql=None):
            """Generate concise 2-3 sentence data insights based on the SQL output."""
            
            system_msg = """You are a data analyst that provides clear, concise insights from data.



RULES:

1. Write ONLY 2-3 sentences maximum

2. Focus on the most important finding

3. Include specific numbers and percentages

4. Use plain, non-technical language

5. Structure as a single paragraph, no bullet points

6. Example format: "[Key finding] increased/decreased by X% to [value], driven by [main factor]. [Additional context if needed]."

Return ONLY the insight text, no markdown or formatting."""

            user_msg = (
                f"User query: {prompt_text}\n"
                f"Data columns: {list(sample_data[0].keys()) if sample_data else 'No data'}\n"
                f"Sample data: {json.dumps(sample_data[:3], default=str) if sample_data else 'No data'}\n"
                f"SQL context: {generated_sql if generated_sql else 'No SQL'}\n\n"
                "Provide a 2-3 sentence insight about the most important trend or finding in this data."
            )

            try:
                llm = get_llm_provider()
                prompt = f"{system_msg}\n\n{user_msg}"
                logger.info("Generating concise insights...")
                insight = llm.generate(prompt).strip()

                # Clean up the response
                insight = insight.strip('"\'')  # Remove any surrounding quotes
                if insight.startswith('Insight: '):
                    insight = insight[9:]  # Remove 'Insight: ' prefix if present
                    
                return insight

            except Exception as e:
                logger.exception("Error generating insights")
                return None



        def generate_charts(prompt_text, sample_data, generated_sql=None):
            """Generate charts using DeepSeek based on the provided prompt and data."""
            system_msg = (f'''You are a data visualization expert that creates beautiful, insightful charts using Plotly.
            IMPORTANT: Use this compact schema for charts. Example:
            [
            {{
                "type": "bar",  // bar, line, pie, scatter, box, etc.
                "title": "Chart Title",
                "x": ["A", "B", "C"],  // X values or categories
                "y": [10, 20, 30],      // Y values
                "y2": [5, 15, 25],      // Optional: Secondary Y values
                "marker_color": "#4285F4",     // Optional: Color for main trace (use marker_color for single color)
                "marker_colors": ["#4285F4", "#EA4335"],  // Optional: Colors for multiple traces or pie segments
                "x_label": "X Axis",    // Optional: X-axis label
                "y_label": "Y Axis",    // Optional: Y-axis label
                "orientation": "v"      // Optional: "h" for horizontal bars
            }}
            ]
            RULES:
            1. ALWAYS return a valid JSON array of chart objects
            2. For bar/line/scatter: 'x' and 'y' are required
            3. For pie: 'labels' and 'values' are required
            4. Keep it minimal - no unnecessary fields
            5. Use simple color codes (hex or named colors)
            6. Max 2 charts per response
            7. Return ONLY the JSON, no markdown code blocks or additional text
            8. VERY IMPORTANT: Use ONLY the values and categories present in the provided sample_data. 
            - Do NOT invent or assume missing categories.
            - If a category is absent in sample_data, simply omit it from the chart.
            - If sample_data has only one category/value, generate a chart with just that category/value.
            ''')
            user_msg = (
                f"User query: {prompt_text}\n"
                f"Available columns: {list(st.session_state.df.columns)}\n"
                f"Sample data (top rows for charting): {json.dumps(sample_data, default=str)}\n"
                f"Context: {'MDL-based SQL used' if generated_sql else 'Row-level similarity context used'}\n\n"
                "Generate at most 2 different visualizations that best represent this data.\n"
                "Return only valid JSON as described."
            )


            try:
                llm = get_llm_provider()
                prompt = f"{system_msg}\n\n{user_msg}"
                print("CHART GENERATION 6677: Prompt:", prompt)
                return llm.generate(prompt).strip()
            except Exception as e:
                logger.exception("Error generating visualization with DeepSeek")
                st.error(f"Failed to generate visualization: {str(e)}")
                return None

        # Generate charts with compact schema
        sample_data = query_result_df.head(12).to_dict(orient='records')
        # sample_data = query_result_df.to_dict(orient='records')
        chart_json = generate_charts(user_prompt, sample_data, generated_sql)
        
        if not chart_json:
            st.warning("The model returned an empty response. Please try rephrasing your query.")
            st.stop()

        st.subheader("Visualized Charts")
        with st.expander("Debug: Raw LLM Response"):
            st.code(chart_json, language="json")
            if generated_sql:
                st.write("### SQL used")
                st.code(generated_sql, language="sql")
        
        # Clean the JSON response
        clean_json = chart_json.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[len("```json"):].strip()
        if clean_json.startswith("```"):
            clean_json = clean_json[3:].strip()
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3].strip()
            
        try:
            # Parse the compact chart config
            import plotly.express as px
            import plotly.graph_objects as go
            

            chart_configs = json.loads(clean_json)
            # Add validation
            if not isinstance(chart_configs, list) or not chart_configs:
                st.error("No valid chart configurations were generated. Please try a different query.")
                st.stop()
            
            for config in chart_configs:
                if not isinstance(config, dict):
                    continue
                chart_type = config.get('type', 'bar')
                title = config.get('title', 'Chart')
                # Skip if required fields are missing
                if chart_type == 'pie':
                    if not all(key in config for key in ['values', 'labels']) or not config['values'] or not config['labels']:
                        st.warning(f"Skipping invalid pie chart config: missing required fields")
                        continue
                else:
                    if not all(key in config for key in ['x', 'y']) or not config['x'] or not config['y']:
                        st.warning(f"Skipping invalid {chart_type} chart config: missing required fields")
                        continue
                if chart_type == 'pie':
                    fig = px.pie(
                        values=config.get('values', []),
                        names=config.get('labels', []),
                        title=title,
                        color_discrete_sequence=[config.get('color', '#4285F4')] if 'color' in config else None
                    )
                else:
                    # Create figure with primary y-axis
                    fig = go.Figure()
                    
                    # Add primary trace
                    if chart_type == 'bar':
                        fig.add_trace(go.Bar(
                            x=config.get('x', []),
                            y=config.get('y', []),
                            name=config.get('y_label', 'Y'),
                            marker_color=config.get('color', '#4285F4'),
                            orientation=config.get('orientation', 'v')
                        ))
                    elif chart_type == 'line':
                        fig.add_trace(go.Scatter(
                            x=config.get('x', []),
                            y=config.get('y', []),
                            name=config.get('y_label', 'Y'),
                            line=dict(color=config.get('color', '#4285F4')),
                            mode='lines+markers'
                        ))
                    else:  # Default to scatter
                        fig.add_trace(go.Scatter(
                            x=config.get('x', []),
                            y=config.get('y', []),
                            name=config.get('y_label', 'Y'),
                            mode='markers',
                            marker=dict(color=config.get('color', '#4285F4'))
                        ))
                    
                    # Add secondary trace if exists
                    if 'y2' in config:
                        fig.add_trace(go.Scatter(
                            x=config.get('x', []),
                            y=config.get('y2'),
                            name=config.get('y2_label', 'Y2'),
                            line=dict(color=config.get('color2', '#EA4335')),
                            yaxis='y2'
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title=title,
                        xaxis_title=config.get('x_label', 'X'),
                        yaxis_title=config.get('y_label', 'Y'),
                        yaxis2={
                            'title': config.get('y2_label', 'Y2'),
                            'overlaying': 'y',
                            'side': 'right',
                            'showgrid': False
                        } if 'y2' in config else None,
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(l=50, r=50, t=50, b=50)
                    )
                
                # Display the figure
                st.plotly_chart(fig, use_container_width=True)
                
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse chart configuration: {str(e)}")
        except Exception as e:
            import traceback
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'chart_json': clean_json if 'clean_json' in locals() else 'Not available',
                'chart_configs': chart_configs if 'chart_configs' in locals() else 'Not available',
                'config': config if 'config' in locals() else 'Not available'
            }
            logger.error("Detailed error information: %s", json.dumps(error_details, indent=2, default=str))
            
            # Show a simplified error to the user
            st.error(f"Error generating visualization: {str(e)}")
            
            # Show more details in an expander for debugging
            with st.expander("Click for error details"):
                st.write("### Error Details")
                st.code(traceback.format_exc())
                
                if 'clean_json' in locals():
                    st.write("### Chart JSON")
                    st.code(clean_json)
                
                if 'chart_configs' in locals():
                    st.write("### Parsed Chart Configs")
                    st.json(chart_configs)
                
                if 'config' in locals():
                    st.write("### Current Chart Config")
                    st.json(config)
          # Generate insights after charts

        st.subheader("📊 Data Insights")

        with st.spinner("Generating insights..."):

            insights = generate_insights(user_prompt, sample_data, generated_sql)

            

            if insights:

                # Display insights as a clean paragraph

                st.markdown("### Key Findings:")

                st.markdown(insights)

                

                # Add insights to expandable section for debugging

                # with st.expander("Debug: Raw Insights Response"):

                #     st.text(insights)

            else:

                st.warning("Could not generate insights. Please try again.")
        

    except RuntimeError as e:
        st.error(str(e))
    except Exception as e:
        logger.exception("Error generating visualization: %s", e)
        st.error(f"Error generating visualization: {e}")