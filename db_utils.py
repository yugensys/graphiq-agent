import os
import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, asdict
import pickle
import logging
from typing import Any, Dict, List, Optional
import os
import sys
import tempfile
from pathlib import Path
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add __module__ to ensure consistent pickling
class Document:
    """A document with content and optional metadata."""
    __module__ = 'db_utils'  # Explicitly set module for pickling
    
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}
    
    def __getstate__(self) -> Dict[str, Any]:
        return {
            'page_content': self.page_content,
            'metadata': self.metadata
        }
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.page_content = state['page_content']
        self.metadata = state.get('metadata', {})
    
    def __repr__(self) -> str:
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            'page_content': self.page_content,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create document from dictionary."""
        return cls(
            page_content=data['page_content'],
            metadata=data.get('metadata', {})
        )

class VectorStore:
    """FAISS-based vector store for document embeddings."""
    
    def __init__(self, dimension: int = 384, index_path: str = "faiss_index"):
        """Initialize the FAISS index and document store."""
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.documents = []
        self.index = None
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or load the FAISS index."""
        if self.index_path.exists():
            self._load_index()
        else:
            # Using FlatL2 for exact search (can be changed to IVFFLAT or HNSW for approximate search)
            self.index = faiss.IndexFlatL2(self.dimension)
        
    def _save_index(self):
        """Save the FAISS index and documents to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path / "index.faiss"))
        
        # Save documents as list of dicts for better compatibility
        documents_data = [doc.to_dict() for doc in self.documents]
        with open(self.index_path / "documents.pkl", "wb") as f:
            pickle.dump(documents_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def add_documents(self, documents: List[Document], model=None):
        """Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            model: Optional model for encoding documents if they don't have embeddings
        """
        if not documents:
            return
            
        try:
            # Ensure index exists
            if self.index is None:
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info("Initialized new FAISS index")
            
            # Convert documents to embeddings if needed
            embeddings = []
            valid_docs = []
            
            for doc in documents:
                try:
                    if hasattr(doc, 'metadata') and 'embedding' in doc.metadata:
                        # Get embedding from metadata if available
                        emb = doc.metadata['embedding']
                        if isinstance(emb, list):
                            emb = np.array(emb, dtype=np.float32)
                        embeddings.append(emb)
                        valid_docs.append(doc)
                    elif model is not None:
                        # Generate embedding using the model
                        emb = model.encode([doc.page_content])[0]
                        embeddings.append(emb)
                        # Store the embedding in metadata for future use
                        if not hasattr(doc, 'metadata') or doc.metadata is None:
                            doc = Document(doc.page_content, {'embedding': emb.tolist()})
                        else:
                            doc.metadata['embedding'] = emb.tolist()
                        valid_docs.append(doc)
                    else:
                        logger.warning("Document has no embedding and no model provided to generate one")
                except Exception as e:
                    logger.error(f"Error processing document: {e}")
            
            if not valid_docs:
                raise ValueError("No valid documents to add")
            
            # Convert to numpy array if needed
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings, dtype=np.float32)
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Store documents
            self.documents.extend(valid_docs)
            
            # Save the updated index
            self._save_index()
            
            logger.info(f"Successfully added {len(valid_docs)} documents to the index")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}", exc_info=True)
            raise   
        
    def _load_index(self):
        """Load the FAISS index and documents from disk if they exist."""
        try:
            if (self.index_path / "index.faiss").exists():
                self.index = faiss.read_index(str(self.index_path / "index.faiss"))
                
                # Load documents if they exist
                if (self.index_path / "documents.pkl").exists():
                    try:
                        with open(self.index_path / "documents.pkl", "rb") as f:
                            documents_data = pickle.load(f)
                            # Convert to Document objects if needed
                            self.documents = []
                            for doc in documents_data:
                                if isinstance(doc, Document):
                                    self.documents.append(doc)
                                elif isinstance(doc, dict):
                                    self.documents.append(Document(
                                        page_content=doc.get('page_content', ''),
                                        metadata=doc.get('metadata', {})
                                    ))
                        logger.info(f"Loaded {len(self.documents)} documents from index")
                    except Exception as e:
                        logger.error(f"Error loading documents: {e}")
                        self.documents = []
                return True
            else:
                # Initialize a new index if none exists
                self.index = faiss.IndexFlatL2(self.dimension)
                self.documents = []
                logger.info("Initialized new FAISS index")
                return False
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            # Fallback to new index on error
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = []
            return False
    
    def similarity_search(
        self, 
        query_embedding: np.ndarray,
        k: int = 4,
        file_hash: Optional[str] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Perform similarity search using FAISS."""
        if not self.documents:
            return []
            
        # Convert query_embedding to numpy array if it's not already
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)
            
        # Reshape for single query
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search the index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:  # Skip invalid indices
                continue
                
            doc = self.documents[idx]
            
            # Get metadata (default to empty dict if not present)
            metadata = getattr(doc, 'metadata', {}) or {}
            doc_file_hash = metadata.get('file_hash')
            
            # Skip if file_hash filter is provided and doesn't match
            if file_hash and doc_file_hash != file_hash:
                continue
                
            # Convert L2 distance to similarity score (1 / (1 + distance))
            distance = float(distances[0][i])
            similarity = 1.0 / (1.0 + distance)
            
            if similarity >= score_threshold:
                results.append({
                    'id': metadata.get('id', ''),
                    'text': getattr(doc, 'page_content', ''),
                    'metadata': metadata,
                    'score': similarity,
                    'file_hash': doc_file_hash
                })
        
        return results

def create_embeddings(
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    file_hash: str,
    model: SentenceTransformer,
    vector_store: 'VectorStore'  # Use string annotation to avoid circular import
):
    """Create and store embeddings using FAISS."""
    try:
        # Generate embeddings using the provided model
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        logger.info("Embeddings generated successfully")
        
        # Normalize embeddings to unit length (important for cosine similarity)
        faiss.normalize_L2(embeddings)
        
        # Create documents with embeddings
        documents = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            # Include the ID in the metadata
            metadata = metadatas[i].copy() if i < len(metadatas) else {}
            metadata.update({
                'id': ids[i],
                'file_hash': file_hash,
                'embedding': embedding  # Store embedding in metadata for now
            })
            
            doc = Document(
                page_content=text,
                metadata=metadata
            )
            documents.append(doc)
        
        # Add to vector store with the model
        vector_store.add_documents(documents, model=model)
        logger.info(f"Successfully stored {len(documents)} embeddings in the vector store")
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}", exc_info=True)
        raise

def reset_database():
    """Reset the vector store by deleting the index directory."""
    import shutil
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
        return True
    return False
def get_or_create_collection(file_hash: str, vector_store: VectorStore) -> bool:
    """Check if a collection (file) exists in the vector store."""
    # In FAISS, we'll just check if we have any documents with this file_hash
    results = vector_store.similarity_search(
        query_embedding=np.zeros(vector_store.dimension),  # Dummy query
        k=30,
        file_hash=file_hash
    )
    return len(results) > 0
