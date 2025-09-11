import os
import requests
from pydantic import Field
import logging
from datetime import datetime
from typing import List, Optional, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

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
    
    if provider == "deepseek":
        return DeepSeekProvider()
    else:
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
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}", exc_info=True)
            raise
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

    return DatasetMDL(
        dataset=dataset_name,
        description=f"Auto-generated schema for {dataset_name}",
        fields=fields,
        constraints=constraints,
    )


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def mdl_to_text(mdl: DatasetMDL) -> str:
    """Convert MDL schema to a formatted text string."""
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

    return "\n".join(lines)


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
        # In a real implementation, you would use a proper vector database
        # For now, return an empty string as we don't have a database
        logger.warning("RAG context retrieval is not implemented without a database")
        return ""
        
    except Exception as e:
        logger.error(f"Error in get_rag_context: {str(e)}")
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

# def generate_sql_query(
#     natural_language_query: str,
#     mdl: DatasetMDL,
#     model_provider: Optional[str] = None,
#     dataset_name: Optional[str] = None,
#     use_rag: bool = False,  
#     top_k: int = 3
# ) -> str:
#     """
#     Convert a natural language query to SQL using MDL schema.

#     Args:
#         natural_language_query: Query string
#         mdl: Dataset schema
#         model_provider: LLM provider
#         dataset_name: Dataset name (unused, kept for backward compatibility)
#         use_rag: Not used, kept for backward compatibility
#         top_k: Not used, kept for backward compatibility

#     Returns:
#         SQL query string
#     """
#     try:
#         # Get the LLM provider
#         llm = get_llm_provider(provider=model_provider)
        
#         # Create prompt with schema
#         prompt = f"""
#         You are an expert SQL generator. 
#         Your task is to create the most appropriate SQL query for a given natural language question.

#         Database schema:
#         {mdl_to_text(mdl)}

#         Guidelines:
#         - Always choose the minimum set of columns needed to answer the question.
#         - If the query asks for a ratio, percentage, distribution, or comparison, 
#         use GROUP BY with aggregation (COUNT, SUM, AVG, etc.).
#         - For "pie chart", "ratio", or "distribution", return grouped counts or proportions.
#         - Do not SELECT all columns unless explicitly requested.
#         - Always alias aggregate columns with meaningful names (e.g., gender_count, total_users).
#         - Use the exact dataset name: "{mdl.dataset}" as the table.

#         Natural language query: {natural_language_query}

#         SQL query:
#         """

#         sql = llm.generate(prompt).strip()
#         resulting_sql=clean_sql(sql, mdl)
#         return resulting_sql
#     except Exception as e:
#         logger.error(f"SQL generation error: {e}")
#         cols = ", ".join([f'"{f.name}"' for f in mdl.fields])
#         return f'SELECT {cols} FROM "{mdl.dataset}" LIMIT 100;'

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

        sql = llm.generate(prompt).strip()

        if sql == "INVALID QUERY":
            logger.error("9999 Invalid query generated")
            return sql  # propagate upwards

        resulting_sql = clean_sql(sql, mdl)
        return resulting_sql

    except Exception as e:
        logger.error(f"SQL generation error: {e}")
        cols = ", ".join([f'"{f.name}"' for f in mdl.fields])
        return f'SELECT {cols} FROM "{mdl.dataset}" LIMIT 100;'

# ---------------------------------------------------------------------------
# CSV to DataFrame
# ---------------------------------------------------------------------------

def process_csv_to_df(csv_content: str) -> pd.DataFrame:
    """Convert CSV content into a pandas DataFrame."""
    from io import StringIO
    try:
        return pd.read_csv(StringIO(csv_content))
    except Exception:
        return pd.read_csv(StringIO(csv_content), engine="python")