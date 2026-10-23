# Arabic Graph RAG

A production-ready, modular **Arabic GraphRAG** system: PDF ingestion → Arabic
cleaning/normalization → Arabic-aware chunking → local embeddings (ChromaDB) →
LLM-built knowledge graph → **Hybrid Retrieval (Dense + BM25 + RRF)** →
**GraphRAG expansion (never bypassed)** → local reranking → contextual
compression → CRAG correction → grounded Arabic answer generation, with a
router (Simple / Basic RAG / Advanced RAG), full centralized JSONL logging,
cost/token tracking, and an LLM-as-judge evaluation pipeline.

All LLM inference runs on **Ollama Cloud** through a single centralized
adapter. Everything else — PDF parsing, cleaning, chunking, embeddings,
vector search, BM25, reranking, and the knowledge graph — runs **100%
locally**, with no GPU or PyTorch required (embeddings/reranking use
ONNX Runtime via `fastembed`).

---

## 1. What's included

```
arabic_graph_rag/
├── config.py                    # Centralized, env-driven configuration
├── .env.example                 # Copy to .env and fill in your Ollama Cloud key
├── requirements.txt
├── data/
│   ├── pdfs/                    # The 3 source Arabic manuals (already included)
│   ├── eval/eval_questions.json # Sample evaluation dataset
│   ├── chroma/                  # Persistent ChromaDB store (created on ingest)
│   ├── graph/                   # Persisted knowledge graph JSON (created on ingest)
│   └── model_cache/             # Local embedding/reranker model cache
├── src/
│   ├── llm/ollama_client.py     # THE single centralized Ollama Cloud adapter
│   ├── ingestion/                # PDF extraction, Arabic cleaning, chunking
│   ├── embeddings/                # Local ONNX embeddings (fastembed)
│   ├── vectorstore/              # Persistent ChromaDB wrapper
│   ├── retrieval/                # BM25, hybrid (RRF) retrieval, reranker, compression
│   ├── graph/                    # Entity/relationship extraction, KG build, expansion
│   ├── rag/                      # Router, query transforms (rewriting, multi-query,
│   │                              # decomposition, HyDE, self-query), CRAG, and the
│   │                              # pipeline orchestrator
│   ├── generation/               # Grounded Arabic answer generation
│   ├── evaluation/               # LLM-as-judge evaluator + CSV/Markdown reports
│   └── utils/                    # Centralized JSONL logger + cost/token tracker
├── scripts/
│   ├── ingest.py                 # PDF -> chunks -> ChromaDB -> knowledge graph
│   ├── build_graph.py            # Rebuild the KG from already-ingested chunks
│   ├── run_query.py              # Ask a question (CLI / interactive REPL)
│   └── run_evaluation.py         # Batch evaluation -> CSV + Markdown report
├── tests/                        # 46 unit/integration tests (pytest)
├── logs/                         # pipeline.jsonl (centralized structured logs)
└── reports/                      # Evaluation CSV + Markdown reports land here
```

### A note on the 3 source PDFs

The three manuals shipped in `data/pdfs/` (`Central_Mail_and_Files_Unit...`,
`Assets_and_wearhouse_operation...`, `Central_Alarm_Tasks_And_Procedures...`)
are **not standard PDF files** — they are ZIP archives (given a `.pdf`
extension) containing one JPEG scan + one pre-OCR'd `.txt` file per page,
plus a `manifest.json` describing page order. `src/ingestion/pdf_extractor.py`
detects this automatically and reads the per-page OCR text directly. Genuine
PDF files are also fully supported (via `pdfplumber`, with a `PyMuPDF`
fallback and an optional `pytesseract` OCR fallback for scanned pages with no
text layer) — you can drop real PDFs into `data/pdfs/` and ingest them the
same way, no configuration changes needed.

---

## 2. Requirements

- Python 3.10+
- An **Ollama Cloud** API key (https://ollama.com → enable Ollama Cloud →
  Settings → API Keys)
- ~2 GB free disk space for the local embedding + reranker model cache
  (downloaded automatically on first run)
- No local LLM, no GPU, no PyTorch install needed

---

## 3. Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# then edit .env and set at minimum:
#   OLLAMA_API_KEY=your_actual_key
#   OLLAMA_MODEL=qwen2.5:latest        # or any other Ollama Cloud model
```

Every model name, path, and retrieval/graph parameter is read from `.env` via
`config.py` — nothing is hard-coded. See `.env.example` for the full list
(embedding model, reranker model, chunk size/overlap, retrieval top-k, RRF
constant, graph expansion hops, CRAG threshold, logging, etc).

---

## 4. Usage

### Step 1 — Ingest the PDFs

Extracts text, cleans/normalizes Arabic, chunks it, embeds it into ChromaDB,
builds the BM25 index, and builds the knowledge graph (entity + relationship
extraction via Ollama Cloud):

```bash
python scripts/ingest.py
```

Useful flags:
```bash
python scripts/ingest.py --reset          # wipe and rebuild the vector store
python scripts/ingest.py --skip-graph     # skip knowledge-graph construction
python scripts/ingest.py --pdf-dir path/to/other/pdfs
```

To only rebuild the knowledge graph (e.g. after changing the entity
extraction prompt or the LLM model) without re-running PDF extraction:

```bash
python scripts/build_graph.py
```

### Step 2 — Ask questions

```bash
python scripts/run_query.py "ما هي خطوات استلام البريد الوارد؟"

# or interactive mode:
python scripts/run_query.py
```

Each run prints the selected route, every technique executed, the model
used, the grounded Arabic answer, its source citations (document + page),
CRAG diagnostics (if triggered), and a token/cost summary.

### Step 3 — Run the evaluation suite

```bash
python scripts/run_evaluation.py
# or with a custom dataset / report label:
python scripts/run_evaluation.py --dataset data/eval/eval_questions.json --label my_run
```

This produces `reports/evaluation_<label>.csv` and
`reports/evaluation_<label>.md` with per-question and aggregate scores for
Context Relevance, Faithfulness, Answer Relevance, and (when a
`reference_answer` is present in the dataset) Correctness.

### Step 4 — Inspect logs

Every pipeline step — route selection, every technique executed, retrieved
chunks, graph expansion, retrieval/reranking scores, tokens, latency, cost,
model used, final answer, evaluation scores, and errors — is written as one
JSON line to `logs/pipeline.jsonl`, so a full run can be reconstructed and
audited:

```bash
tail -f logs/pipeline.jsonl | python3 -m json.tool --json-lines
```

### Step 5 — Optional: chat GUI

A lightweight Streamlit chat interface (`app.py`, project root) wraps the
same `GraphRAGPipeline` used by `scripts/run_query.py` — it does not
reimplement retrieval, routing, or generation. It shows the conversation,
cited sources/pages, and an expandable "Retrieval Details" panel with the
route, techniques executed, graph/rerank/compression counts, token usage,
and cost — all pulled directly from the `PipelineResult` already returned
by `pipeline.answer()`.

```powershell
streamlit run app.py
```

Run this from the project root (same folder as `config.py`). First launch
loads the embedder, BM25 index, and knowledge graph once (same startup
cost as `run_query.py`) and reuses them for the rest of the session. The
sidebar also lists any `reports/*.csv` / `reports/*.md` files already
produced by `scripts/run_evaluation.py`, if present — the GUI only reads
and displays these, it doesn't regenerate or alter them, and "Clear Chat"
only resets the on-screen conversation, not any stored data.

---

## 5. How a query is answered

```
User query
   │
   ▼
Router (Simple / Basic RAG / Advanced RAG)
   │
   ├─ SIMPLE     → direct grounded LLM reply (greetings/meta), no retrieval
   │
   ├─ BASIC RAG  → Hybrid Retrieval (Dense + BM25 + RRF)
   │                     │
   │                     ▼
   │               GraphRAG expansion (ALWAYS applied)
   │                     │
   │                     ▼
   │               Reranking → Contextual Compression → Generation
   │
   └─ ADVANCED   → Self-Query → Query Rewriting → Decomposition
        RAG              │
                          ▼
                    Multi-Query (per sub-question) + HyDE
                          │
                          ▼
                    Hybrid Retrieval for every query variant → RRF fusion
                          │
                          ▼
                    GraphRAG expansion (ALWAYS applied)
                          │
                          ▼
                    CRAG grading → corrective re-retrieval if below threshold
                          │
                          ▼
                    Reranking → Contextual Compression → Generation
```

GraphRAG expansion is invoked from a single shared code path in
`src/rag/pipeline.py` for **both** the Basic and Advanced routes, so it can
never be silently skipped by a route.

---

## 6. Architecture principles followed

- **Single centralized LLM adapter** (`src/llm/ollama_client.py`) — every
  component that needs an LLM call goes through it. No other module talks to
  Ollama Cloud directly.
- **Centralized configuration** — `config.py` + `.env`; no hard-coded model
  names, paths, or thresholds anywhere else in the codebase.
- **Local-only RAG components** — embeddings, ChromaDB, BM25, reranker, and
  the knowledge graph never call an external service; only entity/relation
  extraction and generation call Ollama Cloud (by design, since those need an
  LLM).
- **Centralized JSONL logging** (`src/utils/logger.py`) and **cost/token
  tracking** (`src/utils/cost_tracker.py`), used consistently across every
  module.
- **No duplicated logic** — hybrid retrieval, graph expansion, and JSON-safe
  LLM parsing each live in exactly one place and are reused everywhere they’re
  needed (basic route, advanced route, CRAG’s corrective re-retrieval, etc).
- **Type-hinted, documented, testable** — every public class/function has a
  docstring and type hints; components accept injected dependencies so they
  can be unit-tested without live network/LLM calls.

---

## 7. Testing

```bash
pip install -r requirements.txt   # includes pytest
pytest tests/ -v
```

46 tests covering: Arabic cleaning/normalization, sentence-aware chunking,
the dual-mode PDF extractor (real PDFs + the zip-paginated manual format),
BM25 tokenization/scoring and RRF fusion, knowledge-graph construction/
persistence/traversal, the router's LLM + heuristic-fallback logic, the
Ollama Cloud client (retries, error handling, robust JSON parsing), and full
pipeline orchestration across all three routes (with the reranker/LLM mocked
so the suite runs offline in seconds).

---

## 8. Configuration reference (`.env`)

| Variable | Purpose |
|---|---|
| `OLLAMA_API_KEY`, `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_FAST_MODEL` | Ollama Cloud connection + models (fast model used for routing/rewriting/etc.) |
| `OLLAMA_TIMEOUT`, `OLLAMA_MAX_RETRIES`, `OLLAMA_TEMPERATURE`, `OLLAMA_MAX_TOKENS` | LLM call behavior |
| `COST_PER_1K_PROMPT_TOKENS`, `COST_PER_1K_COMPLETION_TOKENS` | Local cost estimation rates |
| `EMBEDDING_MODEL`, `EMBEDDING_CACHE_DIR` | Local embedding model (ONNX, auto-downloaded) |
| `RERANKER_MODEL`, `RERANKER_CACHE_DIR`, `RERANKER_TOP_K` | Local multilingual cross-encoder reranker |
| `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION_NAME` | Vector store location |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MIN_CHUNK_CHARS` | Chunking behavior |
| `RETRIEVAL_TOP_K`, `DENSE_TOP_K`, `BM25_TOP_K`, `RRF_K`, `HYBRID_ALPHA` | Hybrid retrieval tuning |
| `GRAPH_STORE_DIR`, `GRAPH_STORE_FILE`, `GRAPH_EXPANSION_HOPS`, `GRAPH_MAX_NEIGHBORS`, `GRAPH_MIN_ENTITY_LEN` | GraphRAG tuning |
| `CRAG_RELEVANCE_THRESHOLD`, `CRAG_MAX_CORRECTIVE_QUERIES` | Corrective RAG tuning |
| `LOG_DIR`, `LOG_FILE`, `LOG_LEVEL`, `CONSOLE_LOG` | Logging |
| `REPORTS_DIR`, `EVAL_DATASET` | Evaluation output/input |
| `PDF_DIR` | Source PDF directory |

---

## 9. Troubleshooting

- **"OLLAMA_API_KEY is not set" warning** — copy `.env.example` to `.env` and
  fill in your key; the pipeline will still run but every LLM call will fail
  until it's set.
- **Embedding/reranker model download fails** — these are downloaded from
  Hugging Face on first use and cached under `EMBEDDING_CACHE_DIR` /
  `RERANKER_CACHE_DIR`; make sure the machine running this has outbound
  internet access the first time (subsequent runs use the local cache).
- **Empty/low-quality answers on first query** — make sure you ran
  `python scripts/ingest.py` first; check `data/chroma/` and
  `data/graph/knowledge_graph.json` were populated.
- **OCR fallback not triggering on scanned real PDFs** — install
  `pytesseract` + the Arabic Tesseract language pack (`tesseract-ocr-ara`) and
  `poppler-utils` (required by `pdf2image`) at the OS level; both Python
  packages are already in `requirements.txt` but the underlying OCR engine
  and language data must be installed separately.
