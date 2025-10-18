# rag_backend.py
"""
RAG Agent Backend (FastAPI) — Ollama + FAISS + SQLite
- Auto-ingests files in /mnt/data on startup (skips already-ingested files).
- Detects embedding dim from Ollama on first call and initializes FAISS.
- Persists FAISS index to disk and id_map to a JSON file.
- Exposes endpoints:
    POST /upload   -> upload file and ingest
    POST /query    -> ask query
    GET  /files    -> list ingested files
    GET  /logs     -> ingestion & action logs
- Uses models configured to:
    SLM: llama3.2:3b
    Reasoning: qwen3:4b
    Explain: mistral:latest
    Embeddings: nomic-embed-text:latest
"""
import os, json, time, uuid, threading
from typing import List, Dict, Any, Optional
import sqlite3
import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# try import faiss
import faiss # type: ignore
FAISS_AVAILABLE = True

# CONFIG
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
SLM_MODEL = "llama3.2:3b"
LLM_EXPLAIN_MODEL = "gpt-oss:120b-cloud"
LLM_REASON_MODEL = "deepseek-v3.1:671b-cloud"
EMBED_MODEL = "qwen3-embedding:0.6b"

DATA_DIR = "./data_prototype"
INDEX_DIR = os.path.join(DATA_DIR, "indexes")
META_DB = os.path.join(DATA_DIR, "metadata.db")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
IDMAP_PATH = os.path.join(INDEX_DIR, "id_map.json")
LOGS_TABLE = "logs"
EMBED_DIM: Optional[int] = None

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# sqlite metadata DB
conn = sqlite3.connect(META_DB, check_same_thread=False)
cur = conn.cursor()
cur.execute("""
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
""")
# logs table
cur.execute(f"""
CREATE TABLE IF NOT EXISTS {LOGS_TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL,
    level TEXT,
    message TEXT,
    context TEXT
)
""")
conn.commit()

def log(level: str, message: str, context: Optional[Dict]=None):
    ctx = json.dumps(context) if context else None
    cur.execute(f"INSERT INTO {LOGS_TABLE} (ts, level, message, context) VALUES (?, ?, ?, ?)",
                (time.time(), level, message, ctx))
    conn.commit()
    print(f"[{level}] {message} {ctx or ''}")

# Ollama wrappers
def ollama_embeddings(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    url = f"{OLLAMA_URL}/api/embeddings"
    payload = {"model": model, "input": texts}
    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to decode JSON from Ollama response: {e}. Raw text: {r.text if 'r' in locals() else 'N/A'}")

    # Expected response: { "embedding": [...] } for single text or { "data": [{ "embedding": [...]}, ...] } for multiple texts
    if isinstance(data, dict):
        if "embedding" in data and isinstance(data["embedding"], list):
            # Single embedding response, wrap in a list to match expected return type
            return [data["embedding"]]
        elif "data" in data and isinstance(data["data"], list):
            out = []
            for d in data["data"]:
                emb = d.get("embedding") or d.get("vector")
                if emb and isinstance(emb, list):
                    out.append(emb)
                else:
                    raise RuntimeError(f"Invalid embedding format in 'data' array: {d}")
            return out
    
    raise RuntimeError(f"Unexpected embedding response format from Ollama: {data}")

def ollama_generate(prompt: str, model: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # common shapes
    if isinstance(data, dict):
        if "text" in data:
            return data["text"]
        if "output" in data:
            return data["output"]
        if "choices" in data and len(data["choices"])>0:
            c = data["choices"][0]
            return c.get("text") or c.get("message") or json.dumps(c)
    return str(data)

# Vector index wrapper (initialized once embed dim known)
class VectorIndex:
    def __init__(self, dim:int, path:str):
        self.dim = dim
        self.path = path
        self.id_map: List[str] = []
        self.lock = threading.Lock()
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dim)
        else:
            self.index = None
            self.vectors = []

    def add(self, doc_id: str, vector: np.ndarray):
        with self.lock:
            v = np.asarray(vector, dtype='float32')
            if FAISS_AVAILABLE and self.index is not None:
                self.index.add(np.expand_dims(v, axis=0))
                self.id_map.append(doc_id)
            else:
                self.vectors.append(v)
                self.id_map.append(doc_id)

    def search(self, qvec: np.ndarray, topk: int=5):
        with self.lock:
            qv = np.asarray(qvec, dtype='float32')
            if FAISS_AVAILABLE and self.index is not None:
                D, I = self.index.search(np.expand_dims(qv, axis=0), topk)
                results = []
                for score, idx in zip(D[0], I[0]):
                    if idx < 0 or idx >= len(self.id_map): continue
                    results.append({"doc_id": self.id_map[idx], "score": float(score)})
                return results
            else:
                sims = [float(np.linalg.norm(qv - v)) for v in self.vectors]
                idxs = np.argsort(sims)[:topk]
                return [{"doc_id": self.id_map[int(i)], "score": float(sims[int(i)])} for i in idxs]

    def save(self):
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, self.path)
        # save id_map
        with open(IDMAP_PATH, 'w') as f:
            json.dump(self.id_map, f)

    def load(self):
        if FAISS_AVAILABLE and os.path.exists(self.path):
            self.index = faiss.read_index(self.path)
            # check dim
            if self.index.d != self.dim:
                log("WARN", f"FAISS index dim {self.index.d} != expected {self.dim}")
        if os.path.exists(IDMAP_PATH):
            with open(IDMAP_PATH, 'r') as f:
                self.id_map = json.load(f)
        else:
            self.id_map = []

global_index: Optional[VectorIndex] = None

# Ingestor with header heuristics
class Ingestor:
    def __init__(self):
        pass

    def ingest_file(self, file_path: str, file_name: str) -> str:
        # skip if already ingested file_name exists
        if cur.execute("SELECT 1 FROM documents WHERE file_name = ? LIMIT 1", (file_name,)).fetchone():
            log("INFO", f"File already ingested, skipping: {file_name}")
            return cur.execute("SELECT file_id FROM documents WHERE file_name = ? LIMIT 1", (file_name,)).fetchone()[0]
        file_id = str(uuid.uuid4())
        ext = os.path.splitext(file_name)[1].lower()
        log("INFO", f"Start ingest: {file_name}", {"file": file_name})
        try:
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
            log("INFO", f"Ingest complete: {file_name}", {"file": file_name, "file_id": file_id})
        except Exception as e:
            log("ERROR", f"Ingest failed for {file_name}: {e}", {"file": file_name})
            raise
        return file_id

    def _normalize_sheet_headers(self, sheet_df: pd.DataFrame) -> pd.DataFrame:
        df = sheet_df.copy()
        max_header_rows = min(3, len(df))
        header_rows = []
        for r in range(max_header_rows):
            row_series = df.iloc[r]
            row = row_series.astype(str).tolist()
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
            if df2.empty:
                return pd.DataFrame()
            df2.columns = df2.iloc[0].astype(str).tolist() # Ensure columns are strings
            df2 = df2.iloc[1:].reset_index(drop=True)
            df2.columns = [str(c).strip() if c is not None else f"col_{i}" for i, c in enumerate(df2.columns)]
            return df2

    def _looks_like_text(self, s: str) -> bool:
        s = str(s).strip()
        if s == "": return False
        try:
            float(s.replace(',', ''))
            return False
        except: pass
        if any(ch.isalpha() for ch in s): return True
        return False

    def _ingest_dataframe(self, df: pd.DataFrame, file_id: str, file_name: str):
        df = df.rename(columns=lambda c: str(c).strip())
        # column summaries
        for col in df.columns:
            try:
                col_series = df[col]
                col_vals = col_series.dropna().astype(str).tolist()[:50]
            except Exception as e:
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
                if isinstance(val, pd.Series):
                    # If val is a Series, check if any element is NA
                    sval = "" if val.isna().any() else str(val.tolist()) # Convert to list for string representation
                else:
                    # If val is not a Series, use original logic
                    sval = "" if pd.isna(val) else str(val)
                pairs.append(f"{col}: {sval}")
            excerpt = " | ".join(pairs)
            doc_id = str(uuid.uuid4())
            vecs = ensure_embeddings([excerpt])
            vec = np.array(vecs[0], dtype='float32')
            meta = {"file_id": file_id, "file_name": file_name, "doc_type": "row", "row_id": int(idx), "columns": list(map(str, df.columns))}
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
        cur.execute("INSERT INTO documents (doc_id, file_id, file_name, doc_type, row_id, excerpt, embedding, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (doc_id, file_id, file_name, doc_type, row_id, excerpt, vec.tobytes(), json.dumps(meta), time.time()))
        conn.commit()
        if global_index is None:
            raise RuntimeError("Global index not initialized. ensure_embeddings must be called before ingestion.")
        global_index.add(doc_id, vec)
        # persist index snapshot every N docs? we persist on startup/shutdown/save command
        # For demo we will save after ingestion of each file
        global_index.save()

# ensure embeddings: detect dim and init index
def ensure_embeddings(texts: List[str]) -> List[List[float]]:
    global EMBED_DIM, global_index
    vecs = ollama_embeddings(texts)
    if not vecs: # Check if vecs is empty
        raise RuntimeError("Ollama returned an empty list of embeddings.")
    
    # All embeddings should have the same dimension
    for vec in vecs:
        if not isinstance(vec, list) or not all(isinstance(x, float) for x in vec):
            raise RuntimeError(f"Invalid embedding format received: {vec}. Expected List[float].")

    detected_dim = len(vecs[0])
    if EMBED_DIM is None:
        EMBED_DIM = detected_dim
        global_index = VectorIndex(dim=EMBED_DIM, path=FAISS_INDEX_PATH)
        if os.path.exists(FAISS_INDEX_PATH) and FAISS_AVAILABLE:
            try:
                global_index.load()
                log("INFO", f"Loaded existing FAISS index from {FAISS_INDEX_PATH}")
                # Re-check dimension after loading, in case it was mismatched
                if global_index.index and global_index.index.d != EMBED_DIM:
                    log("WARN", f"Loaded FAISS index dimension {global_index.index.d} mismatches detected embedding dimension {EMBED_DIM}. Re-initializing index.")
                    global_index = VectorIndex(dim=EMBED_DIM, path=FAISS_INDEX_PATH) # Re-initialize if mismatch
            except Exception as e:
                log("WARN", f"Failed to load existing FAISS index: {e}. Initializing new index.")
        else:
            log("INFO", f"Initialized new index with dim={EMBED_DIM}")
    else:
        if EMBED_DIM != detected_dim:
            log("WARN", f"Embedding dimension changed from {EMBED_DIM} to {detected_dim}. This might cause issues.")
    return vecs

# Tools
class Tools:
    @staticmethod
    def filter_table(file_id: str, column: str, op: str, value: Any, limit: int = 100):
        q = "SELECT doc_id, excerpt, metadata FROM documents WHERE file_id = ? AND doc_type = 'row'"
        rows = cur.execute(q, (file_id,)).fetchall()
        results = []
        for doc_id, excerpt, meta_json in rows:
            if str(value).lower() in str(excerpt).lower():
                meta = json.loads(meta_json)
                results.append({"doc_id": doc_id, "excerpt": excerpt, "meta": meta})
                if len(results) >= limit:
                    break
        return {"status": "ok", "rows": results}

    @staticmethod
    def aggregate_sum(file_id: str, column: str, filters: List[Dict[str, Any]] = None):
        q = "SELECT doc_id, excerpt, metadata FROM documents WHERE file_id = ? AND doc_type = 'row'"
        rows = cur.execute(q, (file_id,)).fetchall()
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
                    except: pass
        return {"status": "ok", "metric": "sum", "column": column, "value": total, "count": count}

# Router + Agent
def slm_parse_query_via_ollama(query: str):
    prompt = f"""
You are a strict parser. Given a user's question about tabular/textual data,
output a single JSON object with schema:
{{ "intent": "EXACT_FETCH|FILTER_LIST|AGGREGATE|COMPARE|EXPLAIN|OPINION|UNKNOWN",
  "operation": "...", "metric":"...", "filters":[{{"column":"", "op":"", "value":""}}], "group_by": [], "limit": 100, "confidence": 0.95 }}
Examples:
User: "Show invoice INV-2345"
Output: {{"intent":"EXACT_FETCH","filters":[{{"column":"invoice_id","op":"equals","value":"INV-2345"}}],"confidence":0.98}}
User: "Total sales for Product A in 2023"
Output: {{"intent":"AGGREGATE","operation":"sum","metric":"sales","filters":[{{"column":"product","op":"equals","value":"Product A"}},{{"column":"year","op":"equals","value":2023}}],"confidence":0.95}}
Now parse this query:
\"\"\"{query}\"\"\"\nOnly output JSON.
"""
    raw = ollama_generate(prompt, model=SLM_MODEL, max_tokens=512, temperature=0.0)
    try:
        start = raw.find('{'); end = raw.rfind('}')
        json_str = raw[start:end+1]
        parsed = json.loads(json_str)
    except Exception as e:
        log("WARN", f"SLM parse failed or returned non-JSON. Raw: {raw}", {"err":str(e)})
        parsed = {"intent":"FILTER_LIST", "filters":[], "confidence":0.6}
    return parsed

class Router:
    def decide(self, query: str):
        parsed = slm_parse_query_via_ollama(query)
        intent = parsed.get('intent', 'UNKNOWN')
        if intent == 'EXACT_FETCH': route='SLM_ONLY'
        elif intent == 'AGGREGATE': route='SLM_TO_TOOL'
        elif intent == 'EXPLAIN': route='HYBRID_LLM'
        else: route='SLM_PREFERRED'
        return {"intent": intent, "route": route, "parsed": parsed}

class AgentExecutor:
    def __init__(self):
        self.router = Router()
    def handle_query(self, query: str, prefer_depth: bool=False):
        decision = self.router.decide(query)
        route = decision['route']
        parsed = decision['parsed']
        resp = {"query":query, "intent":decision['intent'], "route":route, "answer_text":"", "structured_output":None, "sources":[], "actions_taken":[], "verifier_status":None, "confidence":parsed.get('confidence',0.5)}
        if route in ('SLM_ONLY','SLM_PREFERRED'):
            filters = parsed.get('filters', [])
            if filters:
                results=[]
                for fid, fname in cur.execute("SELECT DISTINCT file_id, file_name FROM documents").fetchall():
                    for f in filters:
                        r = Tools.filter_table(fid, f.get('column',''), f.get('op','equals'), f.get('value',''))
                        if r['status']=='ok' and len(r['rows'])>0:
                            results.extend(r['rows'])
                resp['structured_output']={"rows":results}; resp['answer_text']=f"Found {len(results)} matching rows."; resp['sources']=results[:5]; resp['verifier_status']='NA'
                return resp
            else:
                qvecs = ensure_embeddings([query]); qvec=np.array(qvecs[0],dtype='float32')
                hits = global_index.search(qvec, topk=6)
                rows=[]
                for h in hits:
                    rec = cur.execute('SELECT doc_id,file_id,file_name,excerpt,metadata FROM documents WHERE doc_id=?',(h['doc_id'],)).fetchone()
                    if rec:
                        doc_id,file_id,file_name,excerpt,meta_json = rec
                        meta=json.loads(meta_json)
                        rows.append({"doc_id":doc_id,"file_id":file_id,"file_name":file_name,"excerpt":excerpt,"score":h['score']})
                resp['structured_output']={"rows":rows}; resp['answer_text']=f"Found {len(rows)} candidate rows via semantic retrieval."; resp['sources']=rows[:5]; resp['verifier_status']='NA'
                return resp
        if route=='SLM_TO_TOOL':
            metric = parsed.get('metric','sales')
            frow = cur.execute('SELECT file_id FROM documents LIMIT 1').fetchone()
            if not frow: resp['answer_text']="No ingested documents yet."; return resp
            file_id = frow[0]
            agg = Tools.aggregate_sum(file_id, metric)
            resp['structured_output']={"aggregate":agg}; resp['answer_text']=f"Aggregate {agg.get('metric')} = {agg.get('value')} (count={agg.get('count')})."; resp['actions_taken']=[{"tool":"aggregate_sum","args":{"file_id":file_id,"column":metric},"result_summary":agg}]; resp['verifier_status']='PASS'
            return resp
        if route=='HYBRID_LLM':
            qvecs = ensure_embeddings([query]); qvec=np.array(qvecs[0],dtype='float32')
            hits = global_index.search(qvec, topk=10)
            contexts=[]
            for h in hits:
                rec = cur.execute('SELECT doc_id,file_id,file_name,excerpt,metadata FROM documents WHERE doc_id=?',(h['doc_id'],)).fetchone()
                if rec:
                    doc_id,file_id,file_name,excerpt,meta_json = rec
                    contexts.append({"doc_id":doc_id,"file_id":file_id,"file_name":file_name,"excerpt":excerpt,"score":h['score']})
            prompt = "You are a data reasoning assistant. Use ONLY the contexts below and any provided aggregates. Do NOT hallucinate.\n\nUser query:\n"
            prompt += query + "\n\nContexts:\n"
            for c in contexts[:8]:
                prompt += f"- {c['excerpt']} (file:{c['file_name']}, doc:{c['doc_id']})\n"
            prompt += "\nAnswer with a short natural-language explanation and then a JSON block with 'structured' results and 'sources'.\n"
            llm_out = ollama_generate(prompt, model=LLM_REASON_MODEL, max_tokens=800, temperature=0.1)
            resp['answer_text']=llm_out; resp['structured_output']={"contexts":contexts[:8]}; resp['sources']=contexts[:8]; resp['verifier_status']='NA'
            return resp
        resp['answer_text']="Unable to route the query."
        return resp

# FastAPI app + startup ingest
from fastapi import FastAPI
app = FastAPI(title="RAG Agent Backend")
ingestor = Ingestor()
agent = AgentExecutor()

class QueryPayload(BaseModel):
    query: str
    prefer_depth: Optional[bool] = False

@app.on_event("startup")
def startup_ingest():
    demo_dir = "/mnt/data"
    if os.path.isdir(demo_dir):
        files = [f for f in os.listdir(demo_dir) if os.path.isfile(os.path.join(demo_dir, f)) and os.path.splitext(f)[1].lower() in ('.csv','.txt','.xls','.xlsx','.pdf','.docx')]
        if files:
            log("INFO", f"Startup ingestion scanning {demo_dir}, found {len(files)} file(s).")
        for fname in files:
            try:
                # check if this file_name already ingested
                if cur.execute("SELECT 1 FROM documents WHERE file_name = ? LIMIT 1", (fname,)).fetchone():
                    log("INFO", f"Skipping already-ingested file {fname}")
                    continue
                path = os.path.join(demo_dir, fname)
                log("INFO", f"Auto-ingesting {path}")
                fid = ingestor.ingest_file(path, fname)
                log("INFO", f"Ingested {fname} -> file_id {fid}")
            except Exception as e:
                log("ERROR", f"Failed to ingest {fname}: {e}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    save_path = os.path.join(DATA_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    file_id = ingestor.ingest_file(save_path, file.filename)
    return {"status":"ok","file_id":file_id}

@app.post("/query")
async def query_endpoint(payload: QueryPayload):
    res = agent.handle_query(payload.query, prefer_depth=payload.prefer_depth)
    # add log
    log("INFO", f"Query executed", {"query": payload.query, "intent": res.get('intent'), "route": res.get('route')})
    return res

@app.get("/files")
async def list_files():
    rows = cur.execute('SELECT DISTINCT file_id, file_name, MIN(created_at) as t FROM documents GROUP BY file_id, file_name').fetchall()
    out=[]
    for fid, fname, t in rows:
        out.append({"file_id":fid,"file_name":fname,"created_at":t})
    return {"files": out}

@app.get("/logs")
def get_logs(limit:int=200):
    rows = cur.execute(f"SELECT ts, level, message, context FROM {LOGS_TABLE} ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    out=[{"ts":r[0],"level":r[1],"message":r[2],"context":json.loads(r[3]) if r[3] else None} for r in rows]
    return {"logs": out}
