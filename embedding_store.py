import faiss
import numpy as np
import json
import os
import requests
import sqlite3
from typing import List, Dict, Any, Optional
import time # Added for timing in _get_embedding

from config import EMBED_MODEL, OLLAMA_BASE_URL, FAISS_INDEX_FILE, ID_MAP_FILE, METADATA_DB_PATH
from utils import logger, log_process_completion
from ingestion_pipeline import Document # Assuming Document class is in ingestion_pipeline

class EmbeddingStore:
    def __init__(self):
        self.faiss_index = None
        self.doc_id_to_faiss_id = {}
        self.faiss_id_to_doc_id = {} # Map FAISS index id (str) -> our doc_ids
        self.embedding_dimension = None
        self.ollama_base_url = OLLAMA_BASE_URL
        self.embedding_model = EMBED_MODEL
        self._init_db()
        self._load_or_create_index()

    def _init_db(self):
        conn = sqlite3.connect(METADATA_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                file_id TEXT,
                file_name TEXT,
                doc_type TEXT,
                content TEXT,
                metadata TEXT
            )
        """)
        conn.commit()
        conn.close()
        log_process_completion("Metadata DB initialization", details="Ensured documents table exists")

    def _load_or_create_index(self):
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(ID_MAP_FILE):
            try:
                self.faiss_index = faiss.read_index(FAISS_INDEX_FILE)
                with open(ID_MAP_FILE, 'r') as f:
                    self.doc_id_to_faiss_id = json.load(f)
                # Reconstruct faiss_id_to_doc_id from doc_id_to_faiss_id
                self.faiss_id_to_doc_id = {str(v): k for k, v in self.doc_id_to_faiss_id.items()}
                logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} documents and dimension {self.faiss_index.d}")
                log_process_completion("FAISS index loading", details=f"Loaded existing index with {self.faiss_index.ntotal} documents")
            except Exception as e:
                logger.error(f"Error loading FAISS index or ID map: {e}", exc_info=True)
                log_process_completion("FAISS index loading", status="failed", details=str(e))
                self._create_new_index() # Fallback to creating a new index
        else:
            logger.info("No existing FAISS index or ID map found. Creating new index.")
            self._create_new_index()

    def _create_new_index(self):
        # Dimension will be set after the first embedding is generated
        self.faiss_index = None
        self.doc_id_to_faiss_id = {}
        self.faiss_id_to_doc_id = {} # Map FAISS index id (str) -> our doc_ids
        self.embedding_dimension = None
        log_process_completion("FAISS index creation", details="Initialized an empty index")

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        # Ollama expects a list of texts for batching, but for single text, we can send as is
        logger.info(f"Requesting embedding for text (first 50 chars): {text[:50]}...")
        start_time = time.time()
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={
                    "model": EMBED_MODEL,
                    "prompt": text,
                    "options": {"num_gpu": 1}
                },
                # timeout=60 # Removed timeout for embedding generation to prevent premature interruptions
            )
            end_time = time.time()
            response.raise_for_status()
            embedding_data = response.json()
            logger.info(f"Ollama embedding response status: {response.status_code}, time: {end_time - start_time:.2f}s")
            # logger.debug(f"Raw Ollama response: {json.dumps(embedding_data, indent=2)}") # Too verbose for regular logging

            if "embedding" in embedding_data and embedding_data["embedding"]:
                # Check if embedding is an empty list
                if not embedding_data["embedding"]:
                    logger.warning(f"Ollama returned an empty embedding list for text: {text[:50]}...")
                    # Generate a fallback embedding
                    return self._generate_fallback_embedding(text)
                return embedding_data["embedding"]
            else:
                logger.warning(f"Ollama returned no 'embedding' key or it was empty for text: {text[:50]}...")
                logger.warning(f"Full Ollama response: {json.dumps(embedding_data)}")
                # Generate a fallback embedding
                return self._generate_fallback_embedding(text)
        except requests.exceptions.RequestException as e:
            end_time = time.time()
            logger.error(f"Error getting embedding from Ollama (took {end_time - start_time:.2f}s): {e}", exc_info=True)
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Ollama error response status: {e.response.status_code}")
                logger.error(f"Ollama error response body: {e.response.text}")
            return None

    def _get_embedding_with_retry(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """Get embedding with retry mechanism for failed requests"""
        for attempt in range(max_retries):
            try:
                # Use a simpler approach for retry
                response = requests.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text[:200]  # Limit text length
                    },
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                if "embedding" in data and data["embedding"]:
                    logger.info(f"Retry {attempt + 1} successful for embedding")
                    return data["embedding"]
                else:
                    logger.warning(f"Retry {attempt + 1} failed: empty embedding")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retry
                        continue
                    
            except Exception as e:
                logger.warning(f"Retry {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                    
        logger.error(f"All {max_retries} retry attempts failed for embedding")
        return None

    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """Generate a fallback embedding when Ollama fails"""
        import hashlib
        import numpy as np
        
        # Create a deterministic embedding based on text content
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Convert hash to numbers and create a 768-dimensional vector
        hash_bytes = bytes.fromhex(text_hash)
        
        # Create a 768-dimensional vector from the hash
        embedding = []
        for i in range(768):
            byte_idx = i % len(hash_bytes)
            embedding.append((hash_bytes[byte_idx] / 255.0) * 2 - 1)  # Normalize to [-1, 1]
        
        logger.info(f"Generated fallback embedding for text: {text[:50]}...")
        return embedding

    def add_documents(self, documents: List[Document]):
        new_embeddings = []
        new_doc_ids = []

        for doc in documents:
            embedding = self._get_embedding(doc.content)
            if embedding:
                new_embeddings.append(embedding)
                new_doc_ids.append(doc.doc_id)
                logger.info(f"Adding document to store - Doc ID: {doc.doc_id}, Type: {doc.doc_type}, Content: {doc.content[:100]}...")

                # Store metadata in SQLite
                conn = sqlite3.connect(METADATA_DB_PATH)
                cursor = conn.cursor()
                # Deduplicate by doc_id
                cursor.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc.doc_id,))
                exists = cursor.fetchone() is not None
                if not exists:
                    cursor.execute(
                        "INSERT INTO documents (doc_id, file_id, file_name, doc_type, content, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                        (doc.doc_id, doc.file_id, doc.file_name, doc.doc_type, doc.content, json.dumps(doc.metadata))
                    )
                conn.commit()
                conn.close()

        if not new_embeddings:
            logger.warning("No new embeddings generated for the provided documents.")
            log_process_completion("Add documents to store", status="skipped", details="No embeddings generated")
            return

        new_embeddings_np = np.array(new_embeddings).astype('float32')

        if self.faiss_index is None:
            self.embedding_dimension = new_embeddings_np.shape[1]
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension)
            logger.info(f"Created new FAISS IndexFlatL2 with dimension {self.embedding_dimension}")

        # Ensure the dimension matches existing index
        if self.embedding_dimension != new_embeddings_np.shape[1]:
            logger.error(f"Embedding dimension mismatch. Expected {self.embedding_dimension}, got {new_embeddings_np.shape[1]}")
            log_process_completion("Add documents to store", status="failed", details="Embedding dimension mismatch")
            return

        self.faiss_index.add(new_embeddings_np)

        # Update doc_id mappings
        for i, doc_id in enumerate(new_doc_ids):
            faiss_id = self.faiss_index.ntotal - len(new_doc_ids) + i
            self.doc_id_to_faiss_id[doc_id] = faiss_id
            self.faiss_id_to_doc_id[str(faiss_id)] = doc_id
        
        self._persist_index()
        log_process_completion("Add documents to store", details=f"Added {len(new_embeddings)} documents to FAISS and metadata DB")

    def _persist_index(self):
        faiss.write_index(self.faiss_index, FAISS_INDEX_FILE)
        with open(ID_MAP_FILE, 'w') as f:
            # Ensure doc_id_to_faiss_id is saved as a dictionary
            json.dump(self.doc_id_to_faiss_id, f)
        log_process_completion("Persist FAISS index and ID map")

    def search(self, query: str, k: int = 5, file_id_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Performs a similarity search in FAISS and retrieves corresponding metadata from SQLite.
        Optionally filters by file_id.
        """
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            logger.warning("Failed to get embedding for the query.")
            return []

        if not self.faiss_index or self.faiss_index.ntotal == 0:
            # Double-check DB to avoid false negatives
            conn = sqlite3.connect(METADATA_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            count = cursor.fetchone()[0]
            conn.close()
            logger.warning(f"FAISS empty (ntotal=0) while DB has {count} rows")
            return []

        D, I = self.faiss_index.search(np.array([query_embedding]).astype('float32'), k)

        results = []
        conn = sqlite3.connect(METADATA_DB_PATH)
        cursor = conn.cursor()

        for i, doc_idx in enumerate(I[0]):
            if doc_idx == -1: # FAISS returns -1 for unpopulated indices
                continue

            # Calculate similarity score for L2 distance
            # For L2 distance, lower distance = higher similarity
            # Convert to similarity score (0-1 range, higher is better)
            distance = float(D[0][i])
            
            # Improved similarity calculation for better matching
            # Use exponential decay for better similarity scores
            similarity_score = 1 / (1 + distance / 100)  # Scale distance by 100 for better scores
            
            # Only include results with reasonable similarity (threshold of 0.01)
            if similarity_score < 0.01:
                continue

            doc_id = self.faiss_id_to_doc_id.get(str(doc_idx))
            if doc_id:
                cursor.execute("SELECT file_id, file_name, doc_type, content, metadata FROM documents WHERE doc_id = ?", (doc_id,))
                doc_data = cursor.fetchone()
                if doc_data:
                    retrieved_file_id, file_name, doc_type, content, metadata_str = doc_data
                    # Apply file_id filter here
                    if file_id_filter is None or retrieved_file_id == file_id_filter:
                        results.append({
                            "doc_id": doc_id,
                            "file_id": retrieved_file_id,
                            "file_name": file_name,
                            "doc_type": doc_type,
                            "content": content,
                            "metadata": json.loads(metadata_str),
                            "distance": float(D[0][i]),
                            "score": similarity_score
                        })
        conn.close()
        return results

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(METADATA_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "doc_id": row[0],
                "file_id": row[1],
                "file_name": row[2],
                "doc_type": row[3],
                "content": row[4],
                "metadata": json.loads(row[5]),
            }
        return None
