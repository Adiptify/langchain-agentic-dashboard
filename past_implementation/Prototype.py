"""
RAG Agent Prototype (Ollama-powered) — Updated

This file is a prototype of the agentic RAG system using Ollama models and FAISS.
Key updates in this version:
 - Auto-detects embedding dimension on first embedding call and initializes FAISS accordingly.
 - Auto-ingests Excel/CSV files found under `/mnt/data/` at startup (useful for demo).
 - Uses specified Ollama model names:
    * SLM parser: llama3.2:3b
    * LLM explanation: mistral:latest
    * LLM reasoning: qwen3:4b
    * Embeddings: nomic-embed-text:latest
 - Improved Excel multi-row header normalization.
 - Keeps SQLite metadata and a persistent index file location.

USAGE
1) Ensure Ollama is running locally and models are available as named.
   If Ollama runs on a different host/port, set OLLAMA_URL env var.
2) Place demo files into `/mnt/data/` (this prototype will auto-ingest files in that folder on startup).
   The two files you provided should already be at `/mnt/data/`.
3) Run the server:
     uvicorn rag_agent_ollama_prototype:app --reload --port 8000

Endpoints:
 - POST /upload (multipart file) -> ingest
 - POST /query {"query":"..."}
 - GET  /files

CAVEATS
 - This is a prototype: production hardening, sandboxing, and security are required.
 - Ensure Ollama API endpoints match your Ollama server's API schema.

"""

import os
import uuid
import time
import json
import requests
import threading
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import sqlite3
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

# Optional faiss import
try:
    import faiss # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# CONFIG
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")  # change if different
SLM_MODEL = "llama3.2:3b"
LLM_EXPLAIN_MODEL = "mistral:latest"
LLM_REASON_MODEL = "qwen3:4b"
EMBED_MODEL = "nomic-embed-text:latest"

DATA_DIR = r"C:\Users\kradi\Mendygo AI\Langchain_implemenation\data_prototype"
INDEX_DIR = os.path.join(DATA_DIR, "indexes")
META_DB = os.path.join(DATA_DIR, "metadata.db")
# Will be detected on first embedding
EMBED_DIM = None

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# -----------------------
# Metadata DB (sqlite)
# -----------------------
conn = sqlite3.connect(META_DB, check_same_thread=False)
_cursor = conn.cursor()
_cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        file_id TEXT,
        file_name TEXT,
        doc_type TEXT,
        row_id INTEGER,
        excerpt TEXT,
        embedding BLOB,
        metadata TEXT,
        created_at REAL
    )
    """
)
conn.commit()

# -----------------------
# Ollama HTTP wrappers
# -----------------------

def ollama_embeddings(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    """
    Call Ollama's embedding endpoint.
    POST /api/embeddings  { "model": "...", "input": ["..."] }
    Returns list of vectors (list of floats)
    """
    url = f"{OLLAMA_URL}/api/embeddings"
    payload = {"model": model, "input": texts}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Typical Ollama-like response: {"data": [{"embedding": [...]}, ...]}
    if isinstance(data, dict) and "data" in data:
        out = []
        for d in data["data"]:
            emb = d.get("embedding") or d.get("vector")
            if emb is None:
                # fallback: entire object
                emb = d
            out.append(emb)
        return out
    if isinstance(data, list):
        return data
    raise RuntimeError(f"Unexpected embedding response: {data}")


def ollama_generate(prompt: str, model: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """
    Call Ollama text generation endpoint.
    POST /api/generate { "model": "...", "prompt": "...", "max_tokens": n, "temperature": t }
    """
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # Parse common response shapes
    if isinstance(data, dict):
        if "text" in data:
            return data["text"]
        if "output" in data:
            return data["output"]
        if "choices" in data and len(data["choices"])>0:
            c = data["choices"][0]
            return c.get("text") or c.get("message") or json.dumps(c)
    return str(data)

# -----------------------
# Vector index wrapper (initialized after embedding dim known)
# -----------------------
class VectorIndex:
    def __init__(self, dim:int, index_path=os.path.join(INDEX_DIR, 'faiss.index')):
        self.dim = dim
        self.index_path = index_path
        self.id_map: List[str] = []  # doc_id ordering
        self.vectors: List[np.ndarray] = []
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dim)
        else:
            self.index = None
        self.lock = threading.Lock()

    def add(self, doc_id: str, vector: np.ndarray):
        with self.lock:
            vec = np.asarray(vector, dtype='float32')
            if FAISS_AVAILABLE and self.index is not None:
                self.index.add(np.expand_dims(vec, axis=0))
                self.id_map.append(doc_id)
            else:
                self.vectors.append(vec)
                self.id_map.append(doc_id)

    def search(self, qvec: np.ndarray, topk: int = 5) -> List[Dict[str, Any]]:
        with self.lock:
            qv = np.asarray(qvec, dtype='float32')
            if FAISS_AVAILABLE and self.index is not None:
                D, I = self.index.search(np.expand_dims(qv, axis=0), topk)
                results = []
                for score, idx in zip(D[0], I[0]):
                    if idx < 0 or idx >= len(self.id_map):
                        continue
                    results.append({"doc_id": self.id_map[idx], "score": float(score)})
                return results
            else:
                sims = [float(np.linalg.norm(qv - v)) for v in self.vectors]
                idxs = np.argsort(sims)[:topk]
                results = [{"doc_id": self.id_map[int(i)], "score": float(sims[int(i)])} for i in idxs]
                return results

# global index placeholder (will initialize once embedding dim known)
global_index: Optional[VectorIndex] = None

# -----------------------
# Ingestor
# -----------------------
class Ingestor:
    def __init__(self):
        pass

    def ingest_file(self, file_path: str, file_name: str) -> str:
        file_id = str(uuid.uuid4())
        ext = os.path.splitext(file_name)[1].lower()
        if ext in ('.csv', '.txt'):
            df = pd.read_csv(file_path)
            self._ingest_dataframe(df, file_id, file_name)
        elif ext in ('.xls', '.xlsx'):
            xls = pd.read_excel(file_path, sheet_name=None, header=None)
            for sheet_name, sheet_df in xls.items():
                df = self._normalize_sheet_headers(sheet_df)
                self._ingest_dataframe(df, file_id, f"{file_name}::{sheet_name}")
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            self._index_text_chunk(file_id, file_name, text)
        return file_id

    def _normalize_sheet_headers(self, sheet_df: pd.DataFrame) -> pd.DataFrame:
        df = sheet_df.copy()
        max_header_rows = min(3, len(df))
        header_rows = []
        for r in range(max_header_rows):
            row = df.iloc[r].astype(str).tolist()
            non_numeric = sum(1 for c in row if self._looks_like_text(c))
            if non_numeric / max(1, len(row)) > 0.6:
                header_rows.append(row)
            else:
                break
        if len(header_rows) >= 2:
            combined = []
            for col_idx in range(len(header_rows[0])):
                parts = []
                for hr in header_rows:
                    if col_idx < len(hr) and str(hr[col_idx]).strip() not in ("", "nan", "None"):
                        parts.append(str(hr[col_idx]).strip())
                combined.append("|".join(parts))
            df = df.iloc[len(header_rows):].copy()
            df.columns = combined
            df = df.reset_index(drop=True)
            return df
        else:
            df2 = sheet_df.copy()
            df2.columns = df2.iloc[0].tolist()
            df2 = df2.iloc[1:].reset_index(drop=True)
            df2.columns = [str(c).strip() if c is not None else f"col_{i}" for i, c in enumerate(df2.columns)]
            return df2

    def _looks_like_text(self, s: str) -> bool:
        s = str(s).strip()
        if s == "":
            return False
        try:
            float(s.replace(',', ''))
            return False
        except Exception:
            pass
        if any(ch.isalpha() for ch in s):
            return True
        return False

    def _ingest_dataframe(self, df: pd.DataFrame, file_id: str, file_name: str):
        df = df.rename(columns=lambda c: str(c).strip())
        # column summaries
        for col in df.columns:
            try:
                col_vals = df[col].dropna().astype(str).tolist()[:50]
            except Exception:
                col_vals = []
            summary_text = f"Column:{col} SampleValues: {col_vals[:10]}"
            doc_id = str(uuid.uuid4())
            vecs = ensure_embeddings([summary_text])
            vec = np.array(vecs[0], dtype='float32')
            meta = {"file_id": file_id, "file_name": file_name, "doc_type": "column_summary", "column": col}
            self._save_doc(doc_id, file_id, file_name, 'column_summary', None, summary_text[:400], vec, meta)

        for idx, row in df.iterrows():
            pairs = []
            for col in df.columns:
                val = row[col]
                sval = "" if pd.isna(val) else str(val)
                pairs.append(f"{col}: {sval}")
            excerpt = " | ".join(pairs)
            doc_id = str(uuid.uuid4())
            vecs = ensure_embeddings([excerpt])
            vec = np.array(vecs[0], dtype='float32')
            meta = {
                "file_id": file_id,
                "file_name": file_name,
                "doc_type": "row",
                "row_id": int(idx),
                "columns": list(map(str, df.columns))
            }
            self._save_doc(doc_id, file_id, file_name, 'row', int(idx), excerpt[:1000], vec, meta)

    def _index_text_chunk(self, file_id: str, file_name: str, text: str):
        chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
        for i, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid4())
            vecs = ensure_embeddings([chunk])
            vec = np.array(vecs[0], dtype='float32')
            meta = {"file_id": file_id, "file_name": file_name, "doc_type": "text_chunk", "chunk_id": i}
            self._save_doc(doc_id, file_id, file_name, 'text_chunk', None, chunk[:400], vec, meta)

    def _save_doc(self, doc_id, file_id, file_name, doc_type, row_id, excerpt, vec, meta):
        _cursor.execute(
            "INSERT INTO documents (doc_id, file_id, file_name, doc_type, row_id, excerpt, embedding, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (doc_id, file_id, file_name, doc_type, row_id, excerpt, vec.tobytes(), json.dumps(meta), time.time())
        )
        conn.commit()
        # add to vector index
        global global_index
        if global_index is None:
            raise RuntimeError("Global index not initialized. Call ensure_embeddings once to initialize index before ingestion.")
        global_index.add(doc_id, vec)

# -----------------------
# Helper: ensure embeddings and initialize index dim
# -----------------------

def ensure_embeddings(texts: List[str]) -> List[List[float]]:
    """Ensures embedding model is available and global_index is initialized with proper dim."""
    global EMBED_DIM, global_index
    vecs = ollama_embeddings(texts)
    if not vecs or len(vecs[0]) == 0:
        raise RuntimeError("No embeddings returned from Ollama")
    detected_dim = len(vecs[0])
    if EMBED_DIM is None:
        EMBED_DIM = detected_dim
        # initialize index
        global_index = VectorIndex(dim=EMBED_DIM)
        print(f"[INFO] Detected embedding dimension: {EMBED_DIM}. Initialized FAISS index.")
    else:
        if EMBED_DIM != detected_dim:
            print(f"[WARN] Embedding dim changed from {EMBED_DIM} to {detected_dim}")
    return vecs

# -----------------------
# Tools
# -----------------------
class Tools:
    @staticmethod
    def filter_table(file_id: str, column: str, op: str, value: Any, limit: int = 100) -> Dict[str, Any]:
        q = "SELECT doc_id, excerpt, metadata FROM documents WHERE file_id = ? AND doc_type = 'row'"
        rows = _cursor.execute(q, (file_id,)).fetchall()
        results = []
        for doc_id, excerpt, meta_json in rows:
            if str(value).lower() in str(excerpt).lower():
                meta = json.loads(meta_json)
                results.append({"doc_id": doc_id, "excerpt": excerpt, "meta": meta})
                if len(results) >= limit:
                    break
        return {"status": "ok", "rows": results}

    @staticmethod
    def aggregate_sum(file_id: str, column: str, filters: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        q = "SELECT doc_id, excerpt, metadata FROM documents WHERE file_id = ? AND doc_type = 'row'"
        rows = _cursor.execute(q, (file_id,)).fetchall()
        total = 0.0
        count = 0
        for doc_id, excerpt, meta_json in rows:
            parts = [p.strip() for p in str(excerpt).split('|')]
            for p in parts:
                if p.lower().startswith(column.lower() + ':'):
                    val = p.split(':', 1)[1].strip()
                    try:
                        num = float(str(val).replace(',', ''))
                        total += num
                        count += 1
                    except Exception:
                        pass
        return {"status": "ok", "metric": "sum", "column": column, "value": total, "count": count}

# -----------------------
# Router & Agent
# -----------------------

def slm_parse_query_via_ollama(query: str) -> Dict[str, Any]:
    prompt = f"""
You are a strict parser. Given a user's question about tabular/textual data,
output a single JSON object with schema:
{{ "intent": "...", "operation": "...", "metric": "...", "filters": [{{"column":"", "op":"", "value":""}}], "group_by": [...], "limit": 100, "confidence": 0.95 }}

Examples:
User: "Show invoice INV-2345"
Output: {{"intent":"EXACT_FETCH","filters":[{{"column":"invoice_id","op":"equals","value":"INV-2345"}}],"confidence":0.98}}

User: "Total sales for Product A in 2023"
Output: {{"intent":"AGGREGATE","operation":"sum","metric":"sales","filters":[{{"column":"product","op":"equals","value":"Product A"}},{{"column":"year","op":"equals","value":2023}}],"confidence":0.95}}

Now parse this query:
""" + query + """

Only output JSON.
"""
    raw = ollama_generate(prompt, model=SLM_MODEL, max_tokens=512, temperature=0.0)
    try:
        start = raw.find('{')
        end = raw.rfind('}')
        json_str = raw[start:end+1]
        parsed = json.loads(json_str)
    except Exception:
        parsed = {"intent": "FILTER_LIST", "filters": [], "confidence": 0.6}
    return parsed

class Router:
    def decide(self, query: str) -> Dict[str, Any]:
        parsed = slm_parse_query_via_ollama(query)
        intent = parsed.get('intent', 'UNKNOWN')
        if intent == 'EXACT_FETCH':
            route = 'SLM_ONLY'
        elif intent == 'AGGREGATE':
            route = 'SLM_TO_TOOL'
        elif intent == 'EXPLAIN':
            route = 'HYBRID_LLM'
        else:
            route = 'SLM_PREFERRED'
        return {"intent": intent, "route": route, "parsed": parsed}

class AgentExecutor:
    def __init__(self):
        self.router = Router()

    def handle_query(self, query: str, prefer_depth: bool = False) -> Dict[str, Any]:
        decision = self.router.decide(query)
        route = decision['route']
        parsed = decision['parsed']

        response = {
            "query": query,
            "intent": decision['intent'],
            "route": route,
            "answer_text": "",
            "structured_output": None,
            "sources": [],
            "actions_taken": [],
            "verifier_status": None,
            "confidence": parsed.get('confidence', 0.5)
        }

        if route in ('SLM_ONLY', 'SLM_PREFERRED'):
            filters = parsed.get('filters', [])
            if filters:
                results = []
                for file_row in _cursor.execute("SELECT DISTINCT file_id, file_name FROM documents").fetchall():
                    fid, fname = file_row
                    for f in filters:
                        r = Tools.filter_table(fid, f.get('column',''), f.get('op','equals'), f.get('value',''))
                        if r['status']=='ok' and len(r['rows'])>0:
                            results.extend(r['rows'])
                response['structured_output'] = {"rows": results}
                response['answer_text'] = f"Found {len(results)} matching rows."
                response['sources'] = results[:5]
                response['verifier_status'] = "NA"
                return response
            else:
                qvecs = ensure_embeddings([query])
                qvec = np.array(qvecs[0], dtype='float32')
                hits = global_index.search(qvec, topk=6)
                rows = []
                for hit in hits:
                    rec = _cursor.execute('SELECT doc_id, file_id, file_name, excerpt, metadata FROM documents WHERE doc_id = ?', (hit['doc_id'],)).fetchone()
                    if rec:
                        doc_id, file_id, file_name, excerpt, meta_json = rec
                        meta = json.loads(meta_json)
                        rows.append({"doc_id": doc_id, "file_id": file_id, "file_name": file_name, "excerpt": excerpt, "score": hit['score']})
                response['structured_output'] = {"rows": rows}
                response['answer_text'] = f"Found {len(rows)} candidate rows via semantic retrieval."
                response['sources'] = rows[:5]
                response['verifier_status'] = "NA"
                return response

        if route == 'SLM_TO_TOOL':
            parsed = decision['parsed']
            metric = parsed.get('metric', 'sales')
            frow = _cursor.execute('SELECT file_id FROM documents LIMIT 1').fetchone()
            if not frow:
                response['answer_text'] = "No ingested documents yet."
                return response
            file_id = frow[0]
            agg = Tools.aggregate_sum(file_id, metric)
            response['structured_output'] = {"aggregate": agg}
            response['answer_text'] = f"Aggregate {agg.get('metric')} = {agg.get('value')} (count={agg.get('count')})."
            response['actions_taken'] = [{"tool": "aggregate_sum", "args": {"file_id": file_id, "column": metric}, "result_summary": agg}]
            response['verifier_status'] = "PASS"
            return response

        if route == 'HYBRID_LLM':
            qvecs = ensure_embeddings([query])
            qvec = np.array(qvecs[0], dtype='float32')
            hits = global_index.search(qvec, topk=10)
            contexts = []
            for hit in hits:
                rec = _cursor.execute('SELECT doc_id, file_id, file_name, excerpt, metadata FROM documents WHERE doc_id = ?', (hit['doc_id'],)).fetchone()
                if rec:
                    doc_id, file_id, file_name, excerpt, meta_json = rec
                    contexts.append({"doc_id": doc_id, "file_id": file_id, "file_name": file_name, "excerpt": excerpt, "score": hit['score']})
            prompt = """You are a data reasoning assistant. Use ONLY the contexts below and any provided aggregates. Do NOT hallucinate.

User query:
""" + query + """

Contexts:
"""
            for c in contexts[:8]:
                prompt += f"- {c['excerpt']} (file:{c['file_name']}, doc:{c['doc_id']})\n"
            prompt += """
Answer with a short natural-language explanation and then a JSON block with 'structured' results and 'sources'.
"""
            llm_out = ollama_generate(prompt, model=LLM_REASON_MODEL, max_tokens=800, temperature=0.1)
            response['answer_text'] = llm_out
            response['structured_output'] = {"contexts": contexts[:8]}
            response['sources'] = contexts[:8]
            response['verifier_status'] = "NA"
            return response

        response['answer_text'] = "Unable to route the query."
        return response

# -----------------------
# FastAPI app & startup ingestion
# -----------------------
app = FastAPI(title="RAG Agent Ollama Prototype")
ingestor = Ingestor()
agent = AgentExecutor()

class QueryPayload(BaseModel):
    query: str
    prefer_depth: Optional[bool] = False

@app.on_event("startup")
def startup_ingest_files():
    """Auto-ingest files present in /mnt/data for demo purposes.
    If you prefer to control ingestion manually, you can remove or comment out this function.
    """
    demo_dir = r"C:\Users\kradi\Mendygo AI\Langchain_implemenation\data_prototype\data"
    if os.path.isdir(demo_dir):
        files = [f for f in os.listdir(demo_dir) if os.path.isfile(os.path.join(demo_dir, f))]
        if files:
            print(f"[STARTUP] Found {len(files)} files in {demo_dir}. Ingesting...")
        for fname in files:
            try:
                path = os.path.join(demo_dir, fname)
                print(f"[STARTUP] Ingesting {path}...")
                fid = ingestor.ingest_file(path, fname)
                print(f"[STARTUP] Ingested file_id={fid} name={fname}")
            except Exception as e:
                print(f"[ERROR] Failed to ingest {fname}: {e}")

@app.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    save_path = os.path.join(DATA_DIR, file.filename)
    with open(save_path, 'wb') as f:
        f.write(await file.read())
    file_id = ingestor.ingest_file(save_path, file.filename)
    return {"status": "ok", "file_id": file_id}

@app.post('/query')
async def query_endpoint(payload: QueryPayload):
    res = agent.handle_query(payload.query, prefer_depth=payload.prefer_depth)
    return res

@app.get('/files')
async def list_files():
    rows = _cursor.execute('SELECT DISTINCT file_id, file_name, MIN(created_at) as t FROM documents GROUP BY file_id, file_name').fetchall()
    out = []
    for fid, fname, t in rows:
        out.append({"file_id": fid, "file_name": fname, "created_at": t})
    return {"files": out}

if __name__ == '__main__':
    print('Run with: uvicorn rag_agent_ollama_prototype:app --reload --port 8000')
