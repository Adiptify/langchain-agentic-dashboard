
System overview (one-line)

A web dashboard backed by a LangChain agentic system that ingests CSV/Excel/PDF/DOCX, normalizes into structured JSON via a small language model (SLM), indexes semantic vectors with Ollama embeddings into FAISS, dynamically routes queries between the SLM and larger LLMs for reasoning or explanation, and executes safe tools (pandas, calculators) to return hybrid answers (natural language + structured JSON + provenance).

# 1. High-level architecture (components & responsibilities)

* **Web UI / Dashboard**

  * Upload files, issue queries, display hybrid results (NL + table + sources), show ingestion status, reindex controls.
* **API Layer (FastAPI / Flask)**

  * Endpoints: `/upload`, `/query`, `/files`, `/reindex`, `/metadata`.
  * Auth, rate-limiting, telemetry hooks.
* **Ingestion Pipeline**
  * Format-specific loaders → normalizers → SLM parser → chunker → embedder → FAISS + metadata store.
* **SLM (Small Language Model)**
  * Structural parser (CSV/Excel normalization, table extraction) and fast JSON output for simple user queries.
* **FAISS (Vector Store)**
  * Stores document vectors and maps doc_id → vector; metadata stored separately in a metadata DB.
* **Metadata DB**
  * SQLite (doc_id → metadata, raw excerpt, file references, numeric fields).
* **LangChain Orchestrator (Agent + Router)**
  * Router (intent detection & policy engine) chooses SLM / LLM / hybrid path and tools; AgentExecutor runs tools and LLMs.
* **Tools**

  * Sandboxed, pre-approved functions: `filter_table`, `aggregate_table`, `describe_column`, `fetch_row`, `search_text`, `download_file`.
* **Verifier**
  * Ensures numeric/factual claims match tool outputs or flagged as “Not available”.
* **Logging & Telemetry**
  * Tracks intent, chosen path, latencies, tool outputs, user feedback.
---

# 2. Ingestion pipeline (from file → searchable knowledge)

## 2.1 Supported formats

* CSV / XLSX (.csv, .xls, .xlsx) — tabular
* PDF (.pdf) — textual, scanned or native
* DOCX (.docx) — textual
* TXT (.txt) — textual

## 2.2 Goals of ingestion
* Preserve exact numeric values for deterministic retrieval and verification.
* Convert messy structure into normalized schema(s).
* Create chunk granularity suitable for row-level retrieval and semantic retrieval.
* Attach rich metadata for filtering and provenance.

## 2.3 Preprocessing steps (format-specific)

**A. CSV / Excel**

1. Load into `pandas.DataFrame`. Preserve sheet names for Excel.
2. Normalize headers:

   * Trim whitespace, force consistent case (e.g., `snake_case`), collapse multi-row headers into a single composite header when necessary.
   * If header rows merged (multi-index), join with delimiter, e.g., `region|country`.
3. Detect column types:

   * Numeric (float, int), date/time, categorical, text.
   * Coerce numeric conversion; if numeric parse fails, create both raw_text and numeric_value fields.
4. Complex/exotic layouts:
   * If the file has pivot-style layout or multi-row headers, run heuristics:
     * If header rows contain date spans, create normalized `date` column.
     * If key columns are repeated across sections, assign a `section_id`.
5. Output:
   * One **row-document per row** (preferred).
   * Additionally, create **column-summary documents** for each column (min, max, mean, unique count, sample values).
6. Metadata for each row-document:
   * `file_id`, `file_name`, `sheet`, `row_id`, `columns` (list), `numeric_fields` (dict), `created_at`, `trusted` flag.
**B. PDF / DOCX / TXT**

1. Extract text (use OCR for scanned PDFs).
2. Identify structural blocks: headings, paragraphs, tables.
3. Chunk text by semantic boundaries: 200–600 tokens with 20–50 token overlap.
4. Create `text_chunk` documents with metadata: `file_id`, `page_no`, `chunk_id`, `section_title`.

## 2.4 SLM structural parsing (core idea)
* For tabular inputs or messy embeddings, the SLM is used to produce **clean JSON** describing table names, columns, and records when heuristics fail.
* Example SLM output schema for an extracted table:



```json
{
  "table_name": "Sales_Q1",
  "columns": [{"name":"region","type":"string"}, {"name":"sales","type":"number"}],
  "records": [
    {"row_id": 1, "region": "North", "sales": 120000},
    {"row_id": 2, "region": "South", "sales": 95000}
  ]
}
```

* SLM output must be validated: ensure numeric fields parse as numbers, ensure `columns` match `records`.

## 2.5 Chunking strategy

* **Tabular**: row-document per row (embedding content = `col1: val | col2: val | ...`).
* **Textual**: 300 tokens per chunk with overlap.
* **Rationale**: Row-level docs yield deterministic fetches (enables quoting row IDs), textual chunks help context for explanations.
---

# 3. Embeddings & FAISS index (how we store and find vectors)

## 3.1 Embedding selection & process

* Use **Ollama embedding** model (local), same model for both docs & queries.
* Pre-normalize text before embedding: strip extraneous whitespace, preserve numeric tokens and column labels.
* For tabular rows, build embedding string that includes column names so embeddings capture structure, e.g., `"date: 2023-01-15 | region: Delhi | sales: 12345.67"`.

## 3.2 FAISS & metadata architecture

* **FAISS** stores vectors + `doc_id` mapping.
* **Metadata DB** (SQLite or Postgres) stores `doc_id -> metadata` and raw excerpt/structured JSON (row values).
* This split allows:

  * Fast vector search via FAISS returning `doc_id`s.
  * Rich filtering and retrieval from metadata DB.

## 3.3 Indexing details & persistence

* Start with `IndexFlatL2` (exact similarity) for small scale.
* For larger scale move to IVF/HNSW + quantization for memory efficiency.
* Persist index file to disk: `faiss_index_v1.index`.
* Persist metadata db snapshot: `documents_v1.parquet` or `sqlite`.
* Versioning: when reindexing, create new version with timestamp. Allow rollback.

## 3.4 Document metadata fields (recommended)

* `doc_id` (uuid)
* `file_id`
* `file_name`
* `doc_type` (`row`, `text_chunk`, `column_summary`)
* `sheet` (if excel)
* `row_id` (if row doc)
* `columns` (list)
* `numeric_fields` (json)
* `text_excerpt`
* `created_at`
* `trusted` boolean
* `ingest_version`

---

# 4. Retriever & Hybrid search (how we find appropriate values)

## 4.1 Retrieval pipeline (steps)

1. **Intent detection** (router): decide whether user requested exact fetch, filter, aggregate, compare, or explanation.
2. If **explicit filters** (date, region, id) present:

   * Use **metadata filtering** with pandas/meta DB to get matching rows. If results ≥ threshold (e.g., ≥1 and ≤ 100), use them directly.
3. Else:

   * Embed query and run FAISS `search(k=K)` — return top-K `doc_ids`.
4. **Re-rank** top-K using hybrid scoring (semantic similarity + metadata match + numeric proximity + recency).
5. If the query asks for aggregate or lists, call `pandas` on filtered rows (tool).
6. If explanation requested, send top-K textual chunks + structured rows to LLM / reasoning engine.

## 4.2 Hybrid scoring formula (example)

`final_score = w_semantic * s_sem + w_meta * s_meta + w_numeric * s_num + w_recency * s_rec`

* Default weights: `w_semantic=0.6`, `w_meta=0.3`, `w_numeric=0.05`, `w_recency=0.05`.
* Adjust per intent: for exact fetchs, raise `w_meta` to 0.7.

## 4.3 Re-ranking using LLM (optional)

* For high-value queries, re-rank top-10 with LLM by asking it to score relevance given the query (costly; use only if required).

---

# 5. Router & Intent detection (decision-making brain)

## 5.1 Intent categories

* `EXACT_FETCH`, `FILTER_LIST`, `AGGREGATE`, `COMPARE`, `EXPLAIN`, `OPINION`, `ACTION`, `UNKNOWN`.

## 5.2 Detection approach

* **Primary**: rule-based + regex (fast and deterministic).

  * e.g., tokens: `invoice`, `order`, `id` → `EXACT_FETCH`; `sum|total|average|mean` → `AGGREGATE`; `why`, `cause`, `explain` → `EXPLAIN`.
* **Secondary**: SLM classifier (if ambiguous). SLM returns `{intent, confidence}` in JSON.
* **Fallback**: if confidence < 0.6, return `AMBIGUOUS` and ask clarifying question.

## 5.3 Router policy mapping (examples)

* `EXACT_FETCH` → SLM parse → metadata filter → return structured result (no LLM).
* `FILTER_LIST` → SLM parse → FAISS + metadata filter → SLM formats results; LLM only if user asks to interpret.
* `AGGREGATE` → SLM parse → run pandas tool → LLM for narrative if requested.
* `EXPLAIN` / `OPINION` → retrieval (top-k) → LLM reasoning model with tool outputs as evidence.

## 5.4 Execution planning

* Router builds a small DAG of actions per query: `Parse -> Retrieve -> ToolCalls -> LLM -> Verify -> Format -> Return`.

---

# 6. Agent tools & safe execution (what the agent can run)
for the coding llm you can use this ollama model glm-4.6:cloud give it the context as such teh desciption fo excel its coloumns and values if the system falls as a fall back approach
## 6.1 Approved tools & interfaces

* `filter_table(table_name, filters, limit)` → returns matching rows JSON.
* `aggregate_table(table_name, group_by, agg_funcs, filters)` → returns aggregate results.
* `describe_column(table_name, column)` → returns distribution summary.
* `fetch_row(file_id, row_id)` → returns raw row and coordinates.
* `search_text(query, k)` → returns top-k textual chunks (from FAISS textual index).
* `download_file(file_id)` → returns file (admin-only).

## 6.2 Tool safety & sandboxing

* Tools run in a sandboxed process/environment with:

  * CPU/memory quotas.
  * No outbound network access.
  * No arbitrary file system writes (only controlled output folder).
  * Timeout (e.g., 10s for heavy ops).
* Tools accept only **validated structured inputs** (no raw user SQL). SLM must produce validated filter JSON before the tool executes.

## 6.3 Tool outputs

* Always return typed JSON:

  * `status` (ok/error)
  * `rows` (list), `aggregate` (object), `meta` (execution time), `query_summary`.
* Agent logs call and associates with original user query id for audit.

---

# 7. SLM vs LLM roles (what each model is responsible for)

## 7.1 SLM (small, fast)

* Parse queries into structured JSON (filters, operations).
* Format retrieval results into compact JSON/tables for UI.
* Handle exact fetch and simple filter/list queries.
* Provide clarifying questions when ambiguous.
* Run first in all pipelines (to produce validated input for tools).

## 7.2 LLM (larger, reasoning)

* Provide in-depth explanations, comparisons, synthesis, and recommendations.
* Use retrieved evidence and tool outputs to generate narratives.
* Re-rank or paraphrase if necessary.
* Ensure it never fabricates: system prompt + verifier ensures claims are grounded.

## 7.3 Model switching rules

* SLM for: `EXACT_FETCH`, `FILTER_LIST`, `AGGREGATE` (calc-only), `AMBIGUOUS` clarifications.
* LLM for: `EXPLAIN`, `OPINION`, complex `COMPARE`, or when user explicitly requests a detailed rationale.
* Hybrid: `AGGREGATE` → run pandas (SLM gives filters) → LLM for explanation if user wants.

---

# 8. Prompting & JSON schemas (practical templates)

## 8.1 SLM parsing prompt (guidelines)

* Provide few-shot examples: user query -> expected JSON parse.
* Ask SLM to **only** output JSON in exact schema.
* Validate SLM output and coerce types (numbers, dates).

**SLM parse schema**:

```json
{
  "intent": "AGGREGATE",
  "operation": "sum",
  "metric": "sales",
  "group_by": ["region"],
  "filters": [{"column":"year","op":"equals","value":2023}],
  "limit": 100,
  "confidence": 0.94
}
```

## 8.2 LLM system prompt (mandatory rules)

* “You are a data assistant. Use ONLY the provided contexts and tool outputs. Do NOT invent values. For each numeric claim, include `file_id` and `doc_id`, or say ‘Not available’.”
* Provide schema of expected final JSON envelope (see section 11).

## 8.3 Output envelope (agent response)

```json
{
  "query": "string",
  "answer_text": "string",
  "structured_output": { /* aggregate/table/list */ },
  "sources": [
    {"file_id":"", "file_name":"", "doc_id":"", "doc_type":"", "excerpt":"", "confidence":0.9}
  ],
  "actions_taken": [
    {"tool": "aggregate_table", "args": {...}, "result_summary": {...}}
  ],
  "verifier_status": "PASS" | "FAIL",
  "confidence": 0.87
}
```

---

# 9. Verification & anti-hallucination (must-haves)

## 9.1 Verifier design

* **Numeric claims**: compare every numeric in LLM text to the tool computed numbers (exact match or tolerance rule for floats).

  * If mismatch: replace with verified value and add note: `"LLM claim inconsistent — using verified data."`
* **Factual claims**: ensure each factual statement points to at least one source in `sources`. If not, flag and remove/ask for clarification.
* **Provenance**: every cited source must include `file_id` and `doc_id` or `row_id`.

## 9.2 Prompt enforcements

* LLM must format final JSON exactly and include `sources`. Use LangChain output parser to reject/repair non-conforming outputs.

---

# 10. UI & Dashboard (web app flow)

## 10.1 Main UI components

* **Uploader**: shows file preview, suggested normalized headers, ingest button.
* **Query box**: example prompts & model preference toggles: `Prefer speed | Prefer depth`.
* **Result panel**:

  * `Natural language` summary (LLM).
  * `Structured table` (from SLM/tool).
  * `Sources` panel with file links and highlighted excerpt.
  * `Action log` showing tools called and verifier status.
* **Admin panel**:

  * Reindex controls, index snapshots, ingestion logs, file management.

## 10.2 User controls & options

* Toggle `Use LLM for explanations` / `Prefer SLM-only` for faster results.
* Option to `Show raw rows` (download row CSV).
* Feedback buttons (thumbs up/down) to collect training labels.

---

# 11. Example execution traces (end-to-end)

## 11.1 Exact fetch — invoice lookup

User: “Show invoice INV-2345.”

1. Preprocessor regex finds `INV-2345`.
2. Router intent → `EXACT_FETCH`.
3. SLM parse returns `{filters: [{"column":"invoice_id", "op":"equals", "value":"INV-2345"}]}` with confidence 0.98.
4. Tool `filter_table` runs; returns 1 row `{file_id: sales_2024.xlsx, row_id: 432, ...}`.
5. SLM formats result JSON table; verifier checks row presence — PASS.
6. Return envelope with `answer_text`, `structured_output` (single row), `sources` (file & row), `confidence: 0.98`.

## 11.2 Aggregate + explain — sales question

User: “What was total sales for product A in 2023 and why did it drop in March?”

1. Router classifies `AGGREGATE` + `EXPLAIN` hybrid.
2. SLM parse gives filters (product=A, year=2023).
3. `aggregate_table` computes sum and monthly deltas.
4. Retriever (FAISS_textual) fetches top-k textual chunks around March and relevant rows.
5. LLM receives aggregates + supporting docs → outputs explanation citing docs and numeric values.
6. Verifier confirms numeric matches aggregate tool — PASS.
7. Return NL summary + structured aggregates + sources + list of rows used.

---

# 12. Index maintenance, reingestion & lifecycle

## 12.1 New file ingestion

* Compute checksum; if new or changed, run ingestion pipeline.
* Create new index version or incrementally add docs to FAISS.
* Save ingestion log.

## 12.2 Rebuild policy

* Schema change (columns added/removed) → full rebuild recommended.
* Small appends → incremental update.

## 12.3 Deletion & redaction

* Mark doc as `deleted` in metadata and remove vector from FAISS (or rebuild).
* Redaction: scrub PII at ingestion or flag for human review.

---

# 13. Monitoring, metrics & testing

## 13.1 Essential metrics

* **Precision@k** (retrieval quality)
* **Answer correctness** (automated checks)
* **Latency** (per model path)
* **Tool failure rate**
* **User satisfaction** (thumbs up/down)
* **Index staleness**

## 13.2 Testing plan

* Unit tests for parser, tooling, retrieval, verifier.
* Integration tests: upload sample dataset, run canonical queries and assert outputs.
* Adversarial tests: malformed queries, ambiguous filters, conflicting conditions.

---

# 14. Security, privacy & governance

## 14.1 Infrastructure

* Ollama + FAISS hosted on company machines; no outbound traffic.
* Auth & RBAC for uploads and dashboards.

## 14.2 Data governance

* PII detection at ingestion (automatic redaction or flag).
* Retention & deletion policies for raw files and vectors.
* Audit logs for every query + tool action.

## 14.3 Access control

* Role-based access (admin, analyst, viewer).
* Fine-grained control for `download_file`.

---

# 15. Operational & scaling considerations

## 15.1 Start small, iterate

* POC: single server with Ollama + FAISS + SQLite, Streamlit for UI.
* Production: move metadata to Postgres, FAISS to sharded/IVF/HNSW or migrate to Milvus/Chroma if needed.

## 15.2 Caching & performance

* Cache frequent aggregate results.
* Cache schema and small embeddings for hot tables.

## 15.3 Cost & latency knobs

* SLM-only for speed (<200 ms).
* LLM for depth: expect seconds (stream partial outputs).
* Use user preference toggles to control LLM usage.

---
Models to be used for 
embedding qwen3-embedding:0.6b
slm  llama3.2:3b
llm reasoning gpt-oss:120b-cloud  
llm    deepseek-v3.1:671b-cloud 