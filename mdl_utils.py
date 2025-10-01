# mdl_utils.py
import os
import requests
from pydantic import Field
import logging
from datetime import datetime
from typing import List, Optional, Any, Tuple
import time
import json


import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import re

# Quick regex patterns used by the local heuristic classifier
GREETINGS_PATTERN = re.compile(
    r"\b(hi|hello|hey|hey there|good morning|good afternoon|good evening)\b", re.I
)
THANKS_PATTERN = re.compile(r"\b(thanks|thank you|thx|ty)\b", re.I)
HELP_PATTERN = re.compile(
    r"\b(help|how do i|what can you do|how to|what can i ask|usage|support)\b", re.I
)

# Use module logger so logs integrate with app logger config
logger = logging.getLogger(__name__)

def get_llm_provider(provider: Optional[str] = None):
    """
    Get an LLM provider instance.

    Args:
        provider: Provider name ('deepseek' or None for default)

    Returns:
        An LLM provider instance with a generate() method
    """
    provider = (provider or "deepseek").lower()
    logger.debug(f"get_llm_provider: selected provider='{provider}'")
    if provider == "deepseek":
        try:
            return DeepSeekProvider()
        except ValueError:
            logger.warning("DeepSeek API key missing — using NoOpLLM fallback.")
            return NoOpLLM()
    raise ValueError(f"Unsupported LLM provider: {provider}")


DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


class DeepSeekProvider:
    """DeepSeek LLM provider via API."""

    def __init__(self, model_name: str = "deepseek-chat"):
        """
        Initialize the DeepSeek API provider.

        Args:
            model_name: Model to use (default: deepseek-chat)
        """
        self.model_name = model_name
        self.api_url = DEEPSEEK_API_URL
        self.api_key = DEEPSEEK_API_KEY
        if not self.api_key:
            logger.error("DeepSeek API key not set. Please export DEEPSEEK_API_KEY.")
            raise ValueError("DeepSeek API key not set. Please export DEEPSEEK_API_KEY.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text using DeepSeek API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature

        Returns:
            Generated text (string)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            t0 = time.time()
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            elapsed_ms = int((time.time() - t0) * 1000)
            # Log at INFO level but avoid logging full prompt (can be large/sensitive)
            logger.info(f"DeepSeek.generate: model={self.model_name} completed time_ms={elapsed_ms}")
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.exception(f"DeepSeek API error: {e}")
            raise

# miniclass fallback 
class NoOpLLM:
    model_name = "noop"
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> str:
        # Be conservative: return clearly invalid responses when prompt expects SQL/JSON
        if "INVALID QUERY" in prompt:
            return "INVALID QUERY"
        # generic useful fallback for summaries / suggestions
        return "LLM unavailable: please set DEEPSEEK_API_KEY to enable advanced features."


# ---------------------------------------------------------------------------
# Schema Models
# ---------------------------------------------------------------------------

class FieldDefinition(BaseModel):
    """Definition of a single dataset field."""
    name: str
    type: str
    description: str = ""
    nullable: bool = True
    format: Optional[str] = None
    enum: Optional[List[Any]] = None
    unit: Optional[str] = None
    example: Optional[Any] = None


class Constraint(BaseModel):
    """Dataset constraint definition."""
    name: str
    type: str
    condition: Optional[str] = None
    columns: Optional[List[str]] = None


class DatasetMDL(BaseModel):
    dataset: str
    description: str = ""
    fields: List[FieldDefinition]
    constraints: List[Constraint] = Field(default_factory=list)  # ✅ safe default
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    version: str = "1.0"

    class Config:
        json_encoders = {
            np.integer: int,
            np.floating: float,
            np.ndarray: lambda v: v.tolist(),
        }


# ---------------------------------------------------------------------------
# Schema Inference
# ---------------------------------------------------------------------------

def infer_field_type(dtype) -> str:
    """Infer field type from pandas dtype."""
    try:
        if pd.api.types.is_integer_dtype(dtype):
            return "integer"
        if pd.api.types.is_float_dtype(dtype):
            return "float"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"
        if pd.api.types.is_bool_dtype(dtype):
            return "boolean"
        if pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            return "string"
    except Exception as e:
        logger.debug(f"infer_field_type caught exception for dtype={dtype}: {e}")
    return "string"


def generate_mdl(df: pd.DataFrame, dataset_name: str) -> DatasetMDL:
    """
    Generate a Model Definition Language (MDL) schema from a pandas DataFrame.

    Args:
        df: Input DataFrame
        dataset_name: Name of the dataset

    Returns:
        DatasetMDL: Generated schema
    """
    t0 = time.time()
    fields: List[FieldDefinition] = []

    for column in df.columns:
        dtype = df[column].dtype
        field_type = infer_field_type(dtype)

        field = FieldDefinition(
            name=column,
            type=field_type,
            description=f"Column {column} of type {field_type}",
            nullable=df[column].isna().any(),
        )

        if field_type == "datetime":
            field.format = "ISO8601"

        unique_values = df[column].dropna().unique()
        if field_type == "string" and len(unique_values) <= 20:
            field.enum = sorted(map(str, unique_values))

        non_null_values = df[column].dropna()
        if not non_null_values.empty:
            field.example = str(non_null_values.iloc[0])

        fields.append(field)

    constraints: List[Constraint] = []

    for col in [c for c in df.columns if "id" in c.lower() or "code" in c.lower()]:
        if df[col].is_unique or df[col].nunique() == len(df):
            constraints.append(Constraint(name=f"{col}_unique", type="unique", columns=[col]))

    for col in df.columns:
        if not df[col].isna().any():
            constraints.append(Constraint(name=f"{col}_not_null", type="not_null", columns=[col]))

    mdl = DatasetMDL(
        dataset=dataset_name,
        description=f"Auto-generated schema for {dataset_name}",
        fields=fields,
        constraints=constraints,
    )

    elapsed_ms = int((time.time() - t0) * 1000)
    logger.info(f"generate_mdl: dataset={dataset_name} rows={len(df)} cols={len(df.columns)} time_ms={elapsed_ms}")
    logger.debug(f"generate_mdl: fields={[f.name for f in fields]}")
    return mdl


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def mdl_to_text(mdl: DatasetMDL) -> str:
    """Convert MDL schema to a formatted text string."""
    try:
        lines = [
            f"Dataset: {mdl.dataset}",
            f"Description: {mdl.description}",
            "\nFields:",
        ]

        for field in mdl.fields:
            desc = f"  - {field.name}: {field.type}"
            if field.description:
                desc += f" - {field.description}"
            desc += " (nullable)" if field.nullable else " (required)"
            if field.enum:
                enum_preview = ", ".join(field.enum[:5])
                if len(field.enum) > 5:
                    enum_preview += f" and {len(field.enum) - 5} more..."
                desc += f" [enum: {enum_preview}]"
            if field.example:
                desc += f" (e.g., {field.example})"
            lines.append(desc)

        if mdl.constraints:
            lines.append("\nConstraints:")
            for c in mdl.constraints:
                if c.type == "unique":
                    lines.append(f"  - {c.name}: Unique on {', '.join(c.columns or [])}")
                elif c.type == "not_null":
                    lines.append(f"  - {c.name}: Not null on {', '.join(c.columns or [])}")
                elif c.condition:
                    lines.append(f"  - {c.name}: {c.type} ({c.condition})")

        text = "\n".join(lines)
        logger.debug(f"mdl_to_text: generated text length={len(text)} for dataset={mdl.dataset}")
        return text
    except Exception as e:
        logger.exception(f"mdl_to_text failed: {e}")
        raise


# ---------------------------------------------------------------------------
# RAG and Embeddings
# ---------------------------------------------------------------------------

def get_rag_context(
    query: str,
    dataset_name: str,
    top_k: int = 3,
    score_threshold: float = 0.7,
    db=None
) -> str:
    """
    Retrieve relevant context using in-memory similarity.

    This is a simplified version that doesn't require a database.
    For a production system, consider using a proper vector database.

    Args:
        query: Query string
        dataset_name: Dataset name (unused in this implementation)
        top_k: Number of results to return
        score_threshold: Minimum similarity score (0-1)
        db: Kept for backward compatibility (unused)

    Returns:
        Context string with relevant information
    """
    try:
        logger.warning("get_rag_context: RAG context retrieval not implemented without a DB")
        return ""

    except Exception as e:
        logger.exception(f"Error in get_rag_context: {str(e)}")
        return ""


# ---------------------------------------------------------------------------
# SQL Query Generation
# ---------------------------------------------------------------------------
def clean_sql(sql: str, mdl: DatasetMDL) -> str:
    # Remove fallback SELECT * queries
    if "SELECT" in sql and "LIMIT 100" in sql and "GROUP BY" not in sql:
        cols = ", ".join([f'"{f.name}"' for f in mdl.fields])
        return f'SELECT {cols} FROM "{mdl.dataset}" LIMIT 100;'
    return sql.strip().rstrip(";") + ";"


def generate_sql_query(
    natural_language_query: str,
    mdl: DatasetMDL,
    model_provider: Optional[str] = None,
    dataset_name: Optional[str] = None,
    use_rag: bool = False,
    top_k: int = 3
) -> str:
    """
    Convert a natural language query to SQL using MDL schema.
    If the query does not align with the dataset schema, return "INVALID QUERY".
    """
    try:
        # Get the LLM provider
        llm = get_llm_provider(provider=model_provider)
        logger.debug(f"generate_sql_query: generating SQL for dataset={dataset_name} query_len={len(natural_language_query)}")

        # Create prompt with schema + strict instructions
        prompt = f"""
You are an expert SQL generator. 
Your task is to create the most appropriate SQL query for a given natural language question.

Database schema:
{mdl_to_text(mdl)}

Guidelines:
- Only generate SQL queries that can be executed on this schema.
- If the natural language query refers to columns, tables, or concepts NOT present in the schema, 
or is ambiguous and cannot be mapped with high confidence, output exactly:
INVALID QUERY
- Always choose the minimum set of columns needed to answer the question.
- If the query asks for a ratio, percentage, distribution, or comparison, 
use GROUP BY with aggregation (COUNT, SUM, AVG, etc.).
- For "pie chart", "ratio", or "distribution", return grouped counts or proportions.
- Do not SELECT all columns unless explicitly requested.
- Always alias aggregate columns with meaningful names (e.g., gender_count, total_users).
- Use the exact dataset name: "{mdl.dataset}" as the table.

Natural language query: {natural_language_query}

Respond with either:
1. A valid SQL query based strictly on the schema, OR
2. The text "INVALID QUERY" (no explanation).
"""

        t0 = time.time()
        sql = llm.generate(prompt).strip()
        gen_ms = int((time.time() - t0) * 1000)
        logger.info(f"generate_sql_query: SQL generated time_ms={gen_ms} provider={getattr(llm, 'model_name', 'unknown')}")
        logger.debug(f"generate_sql_query: sql (trimmed) = {sql[:1000].replace(chr(10),' ')}")

        if sql == "INVALID QUERY":
            logger.error("generate_sql_query: INVALID QUERY returned by LLM")
            return sql

        resulting_sql = clean_sql(sql, mdl)
        return resulting_sql

    except Exception as e:
        logger.exception(f"SQL generation error: {e}")
        cols = ", ".join([f'"{f.name}"' for f in mdl.fields])
        return f'SELECT {cols} FROM "{mdl.dataset}" LIMIT 100;'


# ---------------------------------------------------------------------------
# CSV to DataFrame
# ---------------------------------------------------------------------------

def process_csv_to_df(csv_content: str) -> pd.DataFrame:
    """Convert CSV content into a pandas DataFrame."""
    from io import StringIO
    try:
        df = pd.read_csv(StringIO(csv_content))
        logger.debug(f"process_csv_to_df: parsed csv rows={len(df)} cols={len(df.columns)}")
        return df
    except Exception:
        df = pd.read_csv(StringIO(csv_content), engine="python")
        logger.debug(f"process_csv_to_df (fallback): parsed csv rows={len(df)} cols={len(df.columns)}")
        return df


# ---------------------------------------------------------------------------
# Load dataframe from file
# ---------------------------------------------------------------------------
def load_dataframe_from_file(path: str) -> pd.DataFrame:
    """Load CSV or XLSX into a DataFrame."""
    t0 = time.time()
    try:
        if str(path).endswith(".csv"):
            df = pd.read_csv(path)
        elif str(path).endswith(".xlsx") or str(path).endswith(".xls"):
            df = pd.read_excel(path)
        else:
            raise ValueError("Unsupported file type. Only CSV and XLSX are supported.")
        elapsed_ms = int((time.time() - t0) * 1000)
        logger.info(f"load_dataframe_from_file: loaded {path} rows={len(df)} cols={len(df.columns)} time_ms={elapsed_ms}")
        return df
    except Exception as e:
        logger.exception(f"Failed to load file {path}: {e}")
        raise


# -----------------------------
# New: Local quick-fallback classifier (regex + column matching)
# -----------------------------
def _local_quick_classify(message: str, columns: Optional[List[str]] = None) -> Optional[Tuple[str, float, str]]:
    """
    Fast local heuristics to avoid an LLM call:
      - greetings/help/thanks => CONVERSATION
      - if column tokens appear in message => DATA_QUERY (low confidence)
    Returns (label, confidence, reason) or None if heuristic can't decide.
    """
    msg = (message or "").strip()
    if not msg:
        return ("OUT_OF_SCOPE", 0.99, "empty message")

    if GREETINGS_PATTERN.search(msg) or THANKS_PATTERN.search(msg) or HELP_PATTERN.search(msg):
        return ("CONVERSATION", 0.99, "matched greeting/help/thanks pattern")

    if columns:
        # simple token match: look for exact column names (case-insensitive)
        cols_lower = {c.lower() for c in columns}
        words = re.findall(r"[A-Za-z0-9_]+", msg.lower())
        matched = [w for w in words if w in cols_lower]
        if matched:
            return ("DATA_QUERY", 0.65, f"matched column tokens: {', '.join(sorted(set(matched))) }")

    # unable to decide locally
    return None

# -----------------------------
# New: LLM-based classifier
# -----------------------------

logger = logging.getLogger(__name__)

# Metrics counters (module-level)
LLM_SUCCESS_COUNT = 0
LLM_PARSE_FAIL_COUNT = 0
LLM_ERROR_COUNT = 0

def _extract_json_substring(s: str) -> Optional[str]:
    stack = []
    start_idx = None
    for i, ch in enumerate(s):
        if ch == "{":
            if start_idx is None:
                start_idx = i
            stack.append(ch)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    return s[start_idx:i+1]
    return None

def _llm_classify(message: str, mdl_text: Optional[str], columns: Optional[List[str]]) -> Tuple[str, float, str]:
    global LLM_SUCCESS_COUNT, LLM_PARSE_FAIL_COUNT, LLM_ERROR_COUNT

    schema_snippet = mdl_text if mdl_text else (", ".join(columns) if columns else "No dataset/schema available.")
    base_prompt = f"""
You are a strict classifier. Decide whether the user's message is:
- DATA_QUERY
- CONVERSATION
- OUT_OF_SCOPE

Rules (follow exactly):
1) Use only the information below. Do not produce SQL or extra explanation.
2) Output exactly one JSON object and nothing else, with keys:
   - label: one of "DATA_QUERY","CONVERSATION","OUT_OF_SCOPE"
   - confidence: a float between 0.0 and 1.0
   - reason: one short sentence.
3) The JSON must be the only content in the response. Example:
{{"label":"DATA_QUERY","confidence":0.95,"reason":"matched column 'age'"}}

Dataset/schema (for reference):
{schema_snippet}

User message:
{message}

Now output only the JSON object.
"""

    def _call_llm_and_parse(prompt: str):
        nonlocal message
        try:
            llm = get_llm_provider()
            raw = llm.generate(prompt, temperature=0.0, max_tokens=300)
            raw_str = (raw or "").strip()
            logger.debug(f"Classifier LLM raw output (len={len(raw_str)}): {raw_str[:2000]}")

            # try direct parse
            try:
                parsed = json.loads(raw_str)
            except Exception:
                # try extract {...}
                js = _extract_json_substring(raw_str)
                if js:
                    try:
                        parsed = json.loads(js)
                    except Exception:
                        parsed = None
                else:
                    parsed = None

            if parsed:
                label = parsed.get("label", "").strip()
                confidence = float(parsed.get("confidence", 0.0))
                reason = parsed.get("reason", "")[:1000]
                if label not in {"DATA_QUERY", "CONVERSATION", "OUT_OF_SCOPE"}:
                    logger.warning("LLM returned invalid label: %s", label)
                    return None, raw_str
                return (label, confidence, reason), raw_str

            # parse failed -> return None and raw for logging
            return None, raw_str

        except Exception as e:
            logger.exception("_llm_classify: llm.generate error: %s", e)
            return None, None

    # 1st attempt
    parsed, raw = _call_llm_and_parse(base_prompt)
    if parsed:
        LLM_SUCCESS_COUNT += 1
        return parsed

    # Log raw output (if present) at WARN to make failures searchable
    if raw:
        LLM_PARSE_FAIL_COUNT += 1
        logger.warning("Classifier LLM parse failure. Raw output (first 1000 chars): %s", raw[:1000])
    else:
        LLM_ERROR_COUNT += 1
        logger.warning("Classifier LLM did not return any output or raised an exception.")

    # Retry once with explicit fallback JSON instruction
    retry_prompt = base_prompt + '\nIf you cannot produce EXACT JSON ONLY, return {"label":"OUT_OF_SCOPE","confidence":0.0,"reason":"json_parse_failed"}'
    parsed, raw_retry = _call_llm_and_parse(retry_prompt)
    if parsed:
        LLM_SUCCESS_COUNT += 1
        return parsed

    # still failed -> log both raw attempts (if available)
    if raw_retry:
        logger.warning("Classifier LLM retry parse failure. Raw (retry) (first 1000 chars): %s", raw_retry[:1000])

    # Heuristic fallback: column match if available
    try:
        if columns:
            cols_lower = {c.lower() for c in columns}
            words = re.findall(r"[A-Za-z0-9_]+", message.lower())
            matched = [w for w in words if w in cols_lower]
            if matched:
                return ("DATA_QUERY", 0.6, f"heuristic fallback matched columns: {', '.join(sorted(set(matched)))}")
    except Exception:
        logger.exception("Heuristic fallback failed.")

    return ("OUT_OF_SCOPE", 0.25, "fallback: unable to classify with LLM")


# -----------------------------
# Public API: classify_user_message
# -----------------------------
def classify_user_message(message: str, mdl: Optional[Any] = None, columns: Optional[List[str]] = None) -> Tuple[str, float, str]:
    """
    Classify a user's message relative to the dataset agent.

    Args:
        message: user text
        mdl: optional DatasetMDL object (if present, its text will be used)
        columns: optional list of column names (used for quick heuristics)

    Returns:
        (label, confidence, reason)
    """
    # Ensure columns list if available from mdl
    if columns is None and hasattr(mdl, "fields"):
        try:
            columns = [f.name for f in mdl.fields]
        except Exception:
            columns = None

    # 1) Fast local heuristics (no LLM call)
    try:
        quick = _local_quick_classify(message, columns)
        if quick:
            logger.info(f"classify_user_message: quick-classifier -> {quick}")
            return quick
    except Exception:
        logger.exception("Quick classifier failed; continuing to LLM.")

    # 2) Ask LLM classifier (deterministic)
    mdl_text = None
    try:
        if mdl is not None:
            # if it's a DatasetMDL instance, convert to text representation
            try:
                mdl_text = mdl_to_text(mdl)
            except Exception:
                # if not convertible, fall back to a simple schema text
                try:
                    mdl_text = ", ".join([f.name for f in mdl.fields])
                except Exception:
                    mdl_text = None
    except Exception:
        mdl_text = None

    label, confidence, reason = _llm_classify(message, mdl_text, columns)

    # 3) Post-process: enforce bounds and sensible defaults
    if confidence is None:
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))

    # If confidence below a very low floor, mark as OUT_OF_SCOPE
    if confidence < 0.10 and label != "CONVERSATION":
        logger.warning(f"classifier low confidence {confidence}; downgrading to OUT_OF_SCOPE")
        return ("OUT_OF_SCOPE", confidence, "low confidence fallback")

    logger.info(f"classify_user_message: LLM -> label={label} confidence={confidence} reason={reason}")
    return (label, confidence, reason)

# End of additions to mdl_utils.py