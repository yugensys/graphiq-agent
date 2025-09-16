# db_utils.py
import os
import logging
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import faiss

# Configure logging (single-time)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Document:
    """
    A document with content and optional metadata.
    Metadata should be JSON-serializable friendly (embeddings as lists).
    """

    __module__ = "db_utils"  # helps with pickling across modules

    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __getstate__(self) -> Dict[str, Any]:
        return {"page_content": self.page_content, "metadata": self.metadata}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.page_content = state["page_content"]
        self.metadata = state.get("metadata", {})

    def __repr__(self) -> str:
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary for stable saving."""
        return {"page_content": self.page_content, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(page_content=data.get("page_content", ""), metadata=data.get("metadata", {}) or {})


class VectorStore:
    """
    FAISS-backed vector store that uses cosine similarity semantics via
    IndexFlatIP + normalized vectors.

    - `dimension` should match the embedding dimension.
    - `index_path` is the folder where index.faiss and documents.pkl will be stored.
    """

    def __init__(self, dimension: int = 384, index_path: str = "faiss_index"):
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.documents: List[Document] = []
        self.index: Optional[faiss.Index] = None
        self._initialize_index()

    # ---------------------
    # GPU helper
    # ---------------------
    def _maybe_move_index_to_gpu(self, index: faiss.Index) -> faiss.Index:
        """
        Move a CPU index to GPU if faiss-gpu is available and GPUs exist.
        Returns the index (possibly GPU version). Safe no-op if GPU not usable.
        """
        try:
            ngpu = faiss.get_num_gpus()
            if ngpu > 0:
                logger.info(f"Moving FAISS index to GPU (num_gpus={ngpu})")
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                return gpu_index
        except Exception as e:
            logger.debug("FAISS GPU move not available or failed: %s", e)
        return index

    # ---------------------
    # Index init / save / load
    # ---------------------
    def _initialize_index(self):
        """
        Initialize or load the FAISS index (prefers existing on-disk index).
        """
        try:
            if self.index_path.exists() and (self.index_path / "index.faiss").exists():
                loaded = self._load_index()
                if loaded:
                    return

            # Create a new CPU index (we'll move to GPU if possible)
            cpu_index = faiss.IndexFlatIP(self.dimension)
            self.index = self._maybe_move_index_to_gpu(cpu_index)
            self.documents = []
            logger.info("Initialized new FAISS index (IndexFlatIP)")
        except Exception as e:
            logger.exception("Failed to initialize FAISS index: %s", e)
            raise

    def _save_index(self):
        """Save FAISS index and documents to disk in a stable pickle-friendly format."""
        self.index_path.mkdir(parents=True, exist_ok=True)

        try:
            # If index is on GPU, convert back to CPU for serialization
            index_to_write = self.index
            try:
                if faiss.get_num_gpus() > 0 and index_to_write is not None:
                    index_to_write = faiss.index_gpu_to_cpu(index_to_write)
            except Exception:
                # safe fallback: attempt to write current index (may fail if GPU-only)
                index_to_write = self.index

            faiss.write_index(index_to_write, str(self.index_path / "index.faiss"))
        except Exception as e:
            logger.exception("Failed to write FAISS index: %s", e)
            raise

        # Save documents as list of dicts and ensure embeddings are plain lists
        documents_data = []
        for doc in self.documents:
            meta = dict(doc.metadata or {})
            emb = meta.get("embedding")
            if isinstance(emb, np.ndarray):
                meta["embedding"] = emb.astype(np.float32).tolist()
            documents_data.append({"page_content": doc.page_content, "metadata": meta})

        try:
            with open(self.index_path / "documents.pkl", "wb") as f:
                pickle.dump(documents_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.exception("Failed to write documents.pkl: %s", e)
            raise

    def _load_index(self) -> bool:
        """
        Load FAISS index and saved documents from disk. Returns True on success.
        """
        try:
            idx_file = self.index_path / "index.faiss"
            docs_file = self.index_path / "documents.pkl"

            if idx_file.exists():
                cpu_index = faiss.read_index(str(idx_file))
                self.index = self._maybe_move_index_to_gpu(cpu_index)

                # Load documents if present
                if docs_file.exists():
                    try:
                        with open(docs_file, "rb") as f:
                            documents_data = pickle.load(f)

                        self.documents = []
                        for item in documents_data:
                            if isinstance(item, Document):
                                self.documents.append(item)
                            elif isinstance(item, dict):
                                page_content = item.get("page_content", "")
                                metadata = item.get("metadata", {}) or {}
                                emb = metadata.get("embedding")
                                if isinstance(emb, list):
                                    metadata["embedding"] = np.array(emb, dtype=np.float32)
                                self.documents.append(Document(page_content=page_content, metadata=metadata))

                        logger.info(f"Loaded FAISS index and {len(self.documents)} documents from disk")
                    except Exception as e:
                        logger.exception("Failed to load documents.pkl: %s", e)
                        self.documents = []
                else:
                    self.documents = []

                return True
            else:
                # No on-disk index found, initialize new
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index = self._maybe_move_index_to_gpu(self.index)
                self.documents = []
                logger.info("No FAISS index on disk; initialized new index")
                return False
        except Exception as e:
            logger.exception("Error loading index: %s", e)
            # fallback: new index
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index = self._maybe_move_index_to_gpu(self.index)
            self.documents = []
            return False

    # ---------------------
    # Adding documents
    # ---------------------
    def add_documents(self, documents: List[Document], model=None):
        """
        Add Document objects to the FAISS index.

        If a Document has metadata['embedding'], it's used (converted to np.array if needed).
        Otherwise, if `model` is supplied, it will be used to encode the document text.
        Embeddings are normalized for cosine similarity and stored as float32.
        """
        if not documents:
            return

        try:
            if self.index is None:
                cpu_index = faiss.IndexFlatIP(self.dimension)
                self.index = self._maybe_move_index_to_gpu(cpu_index)
                logger.info("Initialized new FAISS index at add_documents time")

            embeddings_list = []
            docs_to_add: List[Document] = []

            for doc in documents:
                emb_arr = None
                try:
                    meta = getattr(doc, "metadata", {}) or {}

                    # Use provided embedding if present
                    if "embedding" in meta and meta["embedding"] is not None:
                        emb = meta["embedding"]
                        if isinstance(emb, list):
                            emb_arr = np.array(emb, dtype=np.float32)
                        elif isinstance(emb, np.ndarray):
                            emb_arr = emb.astype(np.float32)
                        else:
                            emb_arr = None
                    elif model is not None:
                        # Model expected to have an encode(...) method
                        encoded = model.encode([doc.page_content], show_progress_bar=False)
                        # encoded might be: list-of-arrays or numpy array
                        if isinstance(encoded, list):
                            emb_arr = np.array(encoded[0], dtype=np.float32)
                        else:
                            emb_np = np.array(encoded, dtype=np.float32)
                            emb_arr = emb_np[0] if emb_np.ndim == 2 else emb_np

                        # Persist embedding into metadata as a list for saving
                        if not isinstance(meta, dict):
                            meta = {}
                        meta["embedding"] = emb_arr.tolist()
                        doc.metadata = meta
                    else:
                        logger.warning("Document has no embedding and no model provided: skipping")
                        emb_arr = None
                except Exception as e:
                    logger.exception("Error extracting/creating embedding for a document: %s", e)
                    emb_arr = None

                if emb_arr is None:
                    continue

                # Normalize in-place for inner product = cosine semantics
                faiss.normalize_L2(emb_arr.reshape(1, -1))
                embeddings_list.append(emb_arr.astype(np.float32))

                # Ensure metadata embedding stored as a plain list for persistence
                if isinstance(doc.metadata.get("embedding"), np.ndarray):
                    doc.metadata["embedding"] = doc.metadata["embedding"].astype(np.float32).tolist()

                docs_to_add.append(doc)

            if not docs_to_add:
                raise ValueError("No valid documents to add to index")

            # Stack embeddings to 2D array and add to FAISS
            embeddings_np = np.vstack(embeddings_list).astype(np.float32)
            self.index.add(embeddings_np)

            # Append to in-memory docs list and persist to disk
            self.documents.extend(docs_to_add)
            self._save_index()
            logger.info(f"Added {len(docs_to_add)} documents to FAISS index")
        except Exception as e:
            logger.exception("Error in add_documents: %s", e)
            raise

    # ---------------------
    # Similarity search
    # ---------------------
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 4,
        file_hash: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search the FAISS index for the top-k similar documents.

        Returns a list of dicts with keys: id, text, metadata, score, file_hash

        Score mapping: because we use inner product on normalized vectors, raw scores are in [-1, 1]
        (cosine). We map them to [0, 1] by (score + 1) / 2 for easier thresholds.
        """
        if not self.documents or self.index is None:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        # Normalize query
        faiss.normalize_L2(q)

        try:
            distances, indices = self.index.search(q, k)
        except Exception as e:
            logger.exception("FAISS search failed: %s", e)
            return []

        results: List[Dict[str, Any]] = []
        # distances contain inner product scores if IndexFlatIP; higher is better
        for rank, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.documents):
                logger.warning("Received invalid index from FAISS: %s", idx)
                continue

            try:
                doc = self.documents[idx]
                meta = getattr(doc, "metadata", {}) or {}
                doc_file_hash = meta.get("file_hash")
                # If caller requested a file-specific search, filter by file_hash
                if file_hash and doc_file_hash != file_hash:
                    continue

                raw_score = float(distances[0][rank])
                mapped_score = (raw_score + 1.0) / 2.0  # map [-1,1] -> [0,1]

                if mapped_score >= score_threshold:
                    results.append(
                        {
                            "id": meta.get("id", ""),
                            "text": getattr(doc, "page_content", ""),
                            "metadata": meta,
                            "score": mapped_score,
                            "file_hash": doc_file_hash,
                        }
                    )
            except Exception as e:
                logger.exception("Error processing search result idx=%s: %s", idx, e)

        return results


# -------------------------
# Convenience helpers
# -------------------------
def create_embeddings(
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    file_hash: str,
    model,
    vector_store: "VectorStore",
):
    """
    Create embeddings with a SentenceTransformer-like `model` and store them in vector_store.

    - `model` must implement .encode(list_of_texts, show_progress_bar=False, convert_to_numpy=True)
      (or a compatible return format).
    - This function normalizes embeddings (cosine semantics) and persists them via vector_store.add_documents.
    """
    try:
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize rows for cosine (inner product)
        faiss.normalize_L2(embeddings)

        documents: List[Document] = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            metadata = metadatas[i].copy() if i < len(metadatas) else {}
            metadata.update(
                {
                    "id": ids[i] if i < len(ids) else "",
                    "file_hash": file_hash,
                    "embedding": emb.astype(np.float32).tolist(),
                }
            )
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)

        # Add documents to vector store (model not needed here because embeddings already provided)
        vector_store.add_documents(documents, model=None)
        logger.info(f"Stored {len(documents)} embeddings in vector store")
    except Exception as e:
        logger.exception("Error creating embeddings: %s", e)
        raise


def reset_database(index_path: str = "faiss_index") -> bool:
    """Delete the FAISS index directory (careful!). Returns True if removed."""
    import shutil

    if os.path.exists(index_path):
        shutil.rmtree(index_path)
        return True
    return False


def get_or_create_collection(file_hash: str, vector_store: VectorStore) -> bool:
    """
    Check if any document with the given file_hash exists in the vector store.
    Returns True if found.
    """
    try:
        for d in vector_store.documents:
            meta = getattr(d, "metadata", {}) or {}
            if meta.get("file_hash") == file_hash:
                return True
        return False
    except Exception as e:
        logger.exception("Error checking collection existence: %s", e)
        return False
