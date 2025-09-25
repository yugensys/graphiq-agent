import os
import json
import logging
import numpy as np
from fastapi import FastAPI, HTTPException, Body, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from pathlib import Path
import faiss
import pickle
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import existing utilities
from db_utils import VectorStore, Document
from mdl_utils import DatasetMDL, generate_mdl, get_rag_context, generate_sql_query, clean_sql, get_llm_provider

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure logger is set to DEBUG level

# Initialize FastAPI app
app = FastAPI(
    title="GraphIQ Agent API",
    description="API for processing MDL documents and handling user queries",
    version="1.0.0"
)

# CORS configuration
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
DIMENSION = 384  # Dimension of the embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
VECTOR_STORE = None
EMBEDDING_MODEL_INSTANCE = None

# Initialize the vector store and model
def initialize_services():
    global VECTOR_STORE, EMBEDDING_MODEL_INSTANCE
    try:
        VECTOR_STORE = VectorStore(dimension=DIMENSION, index_path=FAISS_INDEX_PATH)
        EMBEDDING_MODEL_INSTANCE = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Vector store initialized at {FAISS_INDEX_PATH}")
        logger.info(f"Using embedding model: {EMBEDDING_MODEL}")
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        raise

# Call the initialization function
initialize_services()

# Pydantic models for request/response
class MDLInput(BaseModel):
    """Input model for MDL processing"""
    mdl: str = Field(..., description="MDL content as a JSON string")
    sender: str = Field(..., description="Identifier for the sender of the request")
    dataset: str = Field(..., description="Name of the dataset to save the MDL with")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class QueryInput(BaseModel):
    """Input model for query processing"""
    message: str = Field(..., description="User's message or query")
    sender: str = Field(..., description="Identifier for the sender of the message")
    dataset_name: str = Field(default="default_dataset", description="Name of the dataset to query against")
    top_k: Optional[int] = Field(default=3, description="Number of results to return (default: 3)")
    score_threshold: Optional[float] = Field(default=0.7, description="Minimum similarity score (0-1, default: 0.7)")


class QueryResponse(BaseModel):
    """Response model for query results"""
    query: str
    sql_query: Optional[str] = None
    result: Optional[Any] = None
    context: Optional[str] = None
    error: Optional[str] = None

# Helper functions
def _generate_mdl_questions(mdl_text: str, dataset_name: str) -> list[str]:
    """
    Generate potential questions for the given MDL content using LLM.
    
    Args:
        mdl_text: The MDL content as text
        dataset_name: Name of the dataset
        
    Returns:
        List of potential questions as strings
    """
    try:
        # Initialize LLM provider
        llm = get_llm_provider()
        
        # Create prompt for question generation
        prompt = f"""You are an expert data analyst. Given the following MDL (Model Definition Language) 
        for a dataset named '{dataset_name}', generate 3 specific, clear, and relevant questions that 
        a user might ask about this data. The questions should be answerable using the MDL and should 
        demonstrate different types of analysis possible with this data.

        MDL Content:
        {mdl_text}

        Format your response as a JSON array of question strings. Example:
        [
            "What is the distribution of [measure] across [dimension]?",
            "How does [measure] correlate with [another measure]?",
            "What are the top 5 [dimension] by [measure]?"
        ]
        """
        
        # Generate questions using LLM
        response = llm.generate(prompt).strip()
        
        # Parse the response (should be a JSON array of strings)
        try:
            questions = json.loads(response)
            logger.info(f"Generated questions: {questions}")
            if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                return questions[:3]  # Return up to 3 questions
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract questions from plain text
            logger.warning("Failed to parse LLM response as JSON, trying to extract questions from text")
            questions = [q.strip() for q in response.split('\n') if q.strip() and '?' in q]
            return questions[:3] if questions else []
            
        return []
        
    except Exception as e:
        logger.error(f"Error in generate_mdl_questions: {str(e)}")
        return []

# API Endpoints


@app.post("/api/process_mdl", summary="Process MDL document and store embeddings")
async def process_mdl(mdl_input: MDLInput):
    """
    Process an MDL document, generate embeddings, and store them in the vector store.
    """
    try:
        try:
            # Parse the JSON string from the mdl field
            mdl_dict = json.loads(mdl_input.mdl)
            
            # Use the provided dataset name from the request
            dataset_name = mdl_input.dataset
            
            # Update the dataset name in mdl_dict to ensure consistency
            if 'dataset' in mdl_dict:
                mdl_dict['dataset'] = dataset_name
                
            # Convert to MDL object
            mdl = DatasetMDL(**mdl_dict)
            
            # Convert MDL to text for embedding
            mdl_text = mdl_to_text(mdl)
            
            # Log the MDL content before creating the document
            logger.debug("MDL Content to be stored:")
            logger.debug("-" * 40)
            logger.debug(mdl_text)
            logger.debug("-" * 40)
            
            # Create document with metadata
            doc_metadata = {
                "dataset_name": dataset_name,
                "sender": mdl_input.sender,
                "created_at": datetime.utcnow().isoformat(),
                **mdl_input.metadata
            }
            
            logger.debug("Document metadata:")
            logger.debug(json.dumps(doc_metadata, indent=2))
            
            doc = Document(
                page_content=mdl_text,
                metadata=doc_metadata
            )
            
            logger.debug("Document created successfully")
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in mdl field: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )
        
        # Add to vector store
        try:
            logger.debug("Adding document to vector store...")
            VECTOR_STORE.add_documents([doc], model=EMBEDDING_MODEL_INSTANCE)
            logger.debug("Document added to vector store successfully")
            
            # Verify the document was added
            if hasattr(VECTOR_STORE, 'documents'):
                logger.debug(f"Total documents in store after addition: {len(VECTOR_STORE.documents)}")
                if VECTOR_STORE.documents:
                    logger.debug("Last document in store:")
                    last_doc = VECTOR_STORE.documents[-1]
                    logger.debug(f"Type: {type(last_doc)}")
                    if hasattr(last_doc, 'metadata'):
                        logger.debug("Metadata:")
                        logger.debug(json.dumps(last_doc.metadata, indent=2))
        except Exception as e:
            error_msg = f"Error adding document to vector store: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )
        
        # Generate potential questions using LLM
        try:
            questions = _generate_mdl_questions(mdl_text, mdl_input.dataset)
            return {
                "status": "success",
                "message": "MDL processed and stored successfully",
                "dataset": mdl_input.dataset,
                "suggested_questions": questions
            }
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}", exc_info=True)
            return {
                "status": "success",
                "message": "MDL processed and stored successfully, but failed to generate questions",
                "dataset": mdl_input.dataset,
                "suggested_questions": []
            }
    
    except Exception as e:
        logger.error(f"Error processing MDL: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing MDL: {str(e)}"
        )

@app.post("/api/query", response_model=QueryResponse, summary="Process a natural language query")
async def process_query(query_input: QueryInput):
    """
    Process a natural language query against the stored MDL documents.
    """
    try:
        # Get relevant context using RAG
        try:
            context = get_rag_context(
                query=query_input.message, 
                dataset_name=query_input.dataset_name,
                top_k=query_input.top_k,
                score_threshold=query_input.score_threshold
            )
        except Exception as e:
            logger.warning(f"RAG context retrieval warning: {str(e)}")
            context = None
        
        # Generate SQL query
        try:
            logger.info(f"Searching for dataset: {query_input.dataset_name}")
            logger.info(f"Total documents in store: {len(getattr(VECTOR_STORE, 'documents', []))}")
            
            # First try exact match by dataset_name in metadata
            matching_docs = []
            if hasattr(VECTOR_STORE, 'documents'):
                matching_docs = [
                    doc for doc in VECTOR_STORE.documents 
                    if hasattr(doc, 'metadata') and 
                       doc.metadata.get('dataset_name') == query_input.dataset_name
                ]
                logger.info(f"Found {len(matching_docs)} exact matches for dataset: {query_input.dataset_name}")
            
            # If no exact match, try similarity search as fallback
            if not matching_docs:
                logger.info("No exact match found, trying similarity search")
                try:
                    query_embedding = EMBEDDING_MODEL_INSTANCE.encode(query_input.dataset_name)
                    results = VECTOR_STORE.similarity_search(
                        query_embedding=query_embedding,
                        k=5,  # Get more results to increase chance of a match
                        score_threshold=0.7
                    )
                    matching_docs = [r for r in results if hasattr(r, 'page_content')]
                    logger.info(f"Found {len(matching_docs)} similar documents")
                except Exception as e:
                    logger.warning(f"Similarity search failed: {str(e)}")
            
            if not matching_docs:
                # Get list of available dataset names for better error message
                available_datasets = set()
                if hasattr(VECTOR_STORE, 'documents'):
                    available_datasets = {
                        doc.metadata.get('dataset_name') 
                        for doc in VECTOR_STORE.documents 
                        if hasattr(doc, 'metadata')
                    }
                
                return QueryResponse(
                    query=query_input.message,  # Changed from query_input.query
                    error=(
                        f"No MDL found for dataset: '{query_input.dataset_name}'. "
                        f"Available datasets: {', '.join(str(d) for d in available_datasets if d) or 'None'}"
                    )
                )
            
            # Get the most recent document if there are multiple matches
            latest_doc = max(
                matching_docs,
                key=lambda d: d.metadata.get('created_at', '') if hasattr(d, 'metadata') else ''
            )
            
            # Parse the MDL content from the document
            try:
                # Enhanced debug logging
                logger.debug("="*80)
                logger.debug(f"DOCUMENT OBJECT TYPE: {type(latest_doc)}")
                logger.debug(f"DOCUMENT ATTRIBUTES: {dir(latest_doc)}")
                
                # Try different ways to get the content
                if hasattr(latest_doc, 'page_content'):
                    page_content = latest_doc.page_content
                    logger.debug("Got content from page_content attribute")
                elif hasattr(latest_doc, 'content'):
                    page_content = latest_doc.content
                    logger.debug("Got content from content attribute")
                elif hasattr(latest_doc, 'to_dict'):
                    page_content = latest_doc.to_dict()
                    logger.debug("Got content from to_dict() method")
                else:
                    page_content = str(latest_doc)
                    logger.debug("Converted document to string")
                
                if not page_content:
                    raise ValueError("No page content in document")
                
                # Debug logging content
                logger.debug(f"CONTENT TYPE: {type(page_content)}")
                logger.debug("FIRST 200 CHARS OF CONTENT:")
                logger.debug("-" * 40)
                logger.debug(str(page_content)[:200])
                logger.debug("-" * 40)
                
                # Log metadata if available
                if hasattr(latest_doc, 'metadata'):
                    logger.debug("DOCUMENT METADATA:")
                    # logger.debug(json.dumps(latest_doc.metadata, indent=2))
                
                # Try to parse the content
                
                    # First, check if we have a direct MDL dictionary in the document
                    if isinstance(page_content, dict) and 'dataset' in page_content:
                        mdl_dict = page_content
                    else:
                        # Try to parse the text format directly
                        content_str = str(page_content).strip()
                        lines = [line.rstrip() for line in content_str.split('\n')]
                        
                        # Parse dataset name
                        dataset_match = next((line for line in lines if line.startswith('# Dataset:')), None)
                        if not dataset_match:
                            raise ValueError("No dataset found in MDL content")
                            
                        dataset_name = dataset_match.split(':', 1)[1].strip()
                        
                        # Parse description
                        description = ""
                        description_line = next((line for line in lines if line.startswith('Description:')), None)
                        if description_line:
                            description = description_line.split(':', 1)[1].strip()
                        
                        # Parse fields
                        fields = []
                        in_fields_section = False
                        current_field = None
                        
                        for line in lines:
                            line = line.strip()
                            
                            # Check if we're in the fields section
                            if line in ['## Fields:', '## Fields']:
                                in_fields_section = True
                                continue
                                
                            if not in_fields_section:
                                continue
                                
                            # Skip empty lines in fields section
                            if not line:
                                continue
                                
                            # Parse field definition
                            if line.startswith('- ') and '(' in line and ')' in line:
                                # Save previous field if exists
                                if current_field:
                                    fields.append(current_field)
                                
                                # Extract field name and type
                                field_part = line[2:].split('(')
                                field_name = field_part[0].strip()
                                field_type = field_part[1].split(')')[0].strip()
                                
                                current_field = {
                                    'name': field_name,
                                    'type': field_type,
                                    'description': ''
                                }
                            elif current_field:
                                if line.startswith('Description:'):
                                    current_field['description'] = line.split(':', 1)[1].strip()
                                elif line.startswith('Example:'):
                                    current_field['example'] = line.split(':', 1)[1].strip()
                                elif line.startswith('Unit:'):
                                    current_field['unit'] = line.split(':', 1)[1].strip()
                        
                        # Add the last field if exists
                        if current_field and current_field not in fields:
                            fields.append(current_field)
                        
                        # Create the MDL dictionary
                        mdl_dict = {
                            'dataset': dataset_name,
                            'description': description,
                            'fields': fields
                        }
                    
                    logger.debug("Parsed MDL dictionary:")
                    logger.debug(json.dumps(mdl_dict, indent=2))
                    
                    # Convert the dictionary to a DatasetMDL object
                    mdl = DatasetMDL(**mdl_dict)
                    logger.info(f"Successfully parsed MDL for dataset: {getattr(mdl, 'dataset', 'unknown')}")
                    
                
                logger.info(f"Successfully loaded MDL for dataset: {getattr(mdl, 'dataset', 'unknown')}")
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                return QueryResponse(
                    query=query_input.message,  # Changed from query_input.query
                    error=f"Invalid MDL format in stored document: {str(e)}"
                )
                
        except Exception as e:
            logger.error(f"Error retrieving MDL: {str(e)}", exc_info=True)
            return QueryResponse(
                query=query_input.message,  # Changed from query_input.query
                error=f"Error retrieving MDL: {str(e)}"
            )
        
        try:
            # Generate SQL query using the parsed MDL
            sql_query = generate_sql_query(
                natural_language_query=query_input.message,  # Changed from query_input.query
                mdl=mdl,  # Pass the parsed MDL object
                dataset_name=query_input.dataset_name,
                use_rag=bool(context)  # Only use RAG if we have context
            )
            
            # Clean up the SQL query
            if sql_query and sql_query != "INVALID QUERY":
                sql_query = clean_sql(sql_query, mdl)
                
            error_msg = None
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {str(e)}", exc_info=True)
            sql_query = None
            error_msg = str(e)
        
        # Prepare the response
        return QueryResponse(
            query=query_input.message,  # Changed from query_input.query
            sql_query=sql_query,
            context=context,
            result=None,  # You can execute the SQL here if needed
            error=error_msg
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return QueryResponse(
            query=query_input.message if 'query_input' in locals() else "",  # Changed from query_input.query
            error=f"Error processing query: {str(e)}"
        )

# Helper function to convert MDL to text (you may need to adjust this based on your MDL structure)
def mdl_to_text(mdl: DatasetMDL) -> str:
    """Convert MDL to a text representation for embedding."""
    lines = [
        f"# Dataset: {mdl.dataset}",
        f"Description: {mdl.description}",
        "\n## Fields:"
    ]
    
    for field in mdl.fields:
        field_info = [
            f"- {field.name} ({field.type})",
            f"  Description: {field.description}"
        ]
        if field.unit:
            field_info.append(f"  Unit: {field.unit}")
        if field.example is not None:
            field_info.append(f"  Example: {field.example}")
            
        lines.append("\n".join(field_info))
    
    if mdl.constraints:
        lines.append("\n## Constraints:")
        for constraint in mdl.constraints:
            lines.append(f"- {constraint.name} ({constraint.type}): {constraint.condition or ''}")
    
    return "\n".join(lines)

# Health check endpoint
@app.get("/health", include_in_schema=False)
async def health_check():
    """
    Health check endpoint for Docker and load balancers.
    Returns 200 if all services are healthy, 500 otherwise.
    """
    try:
        # Basic service status
        status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "fastapi": "running",
                "vector_store": "initialized" if VECTOR_STORE is not None else "not_initialized",
                "embedding_model": "loaded" if EMBEDDING_MODEL_INSTANCE is not None else "not_loaded"
            }
        }
        
        # Check vector store health
        if VECTOR_STORE is None:
            raise Exception("Vector store not initialized")
            
        # Check embedding model health
        if EMBEDDING_MODEL_INSTANCE is None:
            raise Exception("Embedding model not loaded")
            
        return status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# Main entry point for running the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
