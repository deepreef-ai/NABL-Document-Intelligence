# Deployment & Configuration — Agentic Extraction Workflow

Companion to `GRAPH_ARCHITECTURE.md`. Everything needed to run the workflow
locally, in Docker, and in production.

---

## 1. Local

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate          # source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cp .env.example .env            # then set at least one provider (below)
uvicorn app.main:app --reload --port 8000
```

Verify:

```bash
curl localhost:8000/graph/health
curl "localhost:8000/graph/health?check_llm=true"   # also probes the providers
```

Run the offline sample — no credentials needed, exercises the real graph with a
scripted model:

```bash
python scripts/graph_sample_run.py
```

Run the tests:

```bash
pytest tests/graph -q                        # 83, all offline
pytest -q --ignore=tests/test_benchmark.py   # with the existing suite
```

---

## 2. Docker

```bash
docker compose up --build          # api + postgres
docker compose --profile ui up     # also the frontend
```

The image is multi-stage: compilers live in the build stage only. It runs as
UID 10001 and owns just its storage directories. The healthcheck hits
`/graph/health`, so a container that is listening but cannot reach its database
reports unhealthy rather than green.

**One uvicorn worker on purpose.** The extraction node already runs a thread
pool per document; multiple workers multiply concurrent LLM calls and blow
through the provider rate limit, which is per-account and not per-process.
Scale with replicas plus a shared Postgres checkpointer.

---

## 3. Environment variables

### Providers

| Variable | Default | Notes |
|---|---|---|
| `GRAPH_PROVIDER_ORDER` | `nova,gemini` | ordered fallback chain; Nova primary per spec |
| `NOVA_MODEL` | `us.amazon.nova-2-lite-v1:0` | must be an **inference-profile** id (`us.` prefix) — Nova 2 Lite does not support plain on-demand |
| `NOVA_REGION` | `us-east-1` | |
| `NOVA_MAX_TOKENS` | `8192` | output cap; the binding limit on big chunks |
| `GEMINI_API_KEY` | — | |
| `GEMINI_MODEL` | `gemini-3.6-flash` | |
| `LLM_TIMEOUT_SECONDS` | `180` | a 269-row table needs real headroom |

Nova authenticates via the **ambient AWS credential chain**, not an API key —
instance role, `AWS_PROFILE`, or `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`.
The identity needs `bedrock:InvokeModel` on the model **and** model access
granted in the Bedrock console for that region.

### Retry, rate limiting, concurrency

| Variable | Default |
|---|---|
| `MAX_ATTEMPTS_PER_CHUNK` | `3` |
| `MAX_EXTRACTION_ROUNDS` | `3` |
| `MAX_REANALYSIS_ROUNDS` | `2` |
| `BACKOFF_BASE_SECONDS` / `BACKOFF_MAX_SECONDS` | `2` / `60` |
| `MAX_REQUESTS_PER_MINUTE` | `60` |
| `MAX_CONCURRENT_LLM_CALLS` | `4` |

### Chunking and OCR

| Variable | Default | Notes |
|---|---|---|
| `MAX_CHUNK_CHARS` | `24000` | raising this cuts calls but risks truncated replies |
| `CHUNK_OVERLAP_PAGES` | `1` | context only; overlap pages are not claimed as covered |
| `MAX_FIELDS_PER_CHUNK` | `120` | caps the reply size |
| `OCR_ENABLED` | `true` | |
| `OCR_DPI` | `200` | |

### Thresholds

| Variable | Default | Meaning |
|---|---|---|
| `VERIFIED_CONFIDENCE_THRESHOLD` | `0.75` | promote UNCERTAIN → VERIFIED when evidence supports it |
| `EVIDENCE_SIMILARITY_THRESHOLD` | `0.88` | fuzzy quote matching |
| `MAPPING_CONFIDENCE_THRESHOLD` | `0.70` | below this, MAPPED downgrades to PARTIALLY_MAPPED |
| `LOW_CONFIDENCE_RATIO_LIMIT` | `0.30` | above this unverified share → `LOW_CONFIDENCE` |

### Limits, storage, behaviour

| Variable | Default |
|---|---|
| `MAX_FILE_SIZE_MB` / `MAX_PAGES` | `200` / `2000` |
| `WORKFLOW_TIMEOUT_SECONDS` | `3600` |
| `GRAPH_DATABASE_URL` | *(blank → SQLite)* |
| `GRAPH_SQLITE_PATH` | `graph_storage/graph.db` |
| `CHECKPOINT_DB_PATH` | `graph_storage/checkpoints.db` |
| `GRAPH_PIPELINE_ENABLED` | `true` — **rollback lever**; false restores the legacy path |
| `HUMAN_REVIEW_ENABLED` | `true` |
| `INTERRUPT_FOR_HUMAN_REVIEW` | `false` — true pauses the graph on unresolved conflicts |

---

## 4. Moving to Postgres

Two steps, and only one file changes:

```bash
pip install langgraph-checkpoint-postgres
export GRAPH_DATABASE_URL=postgresql://user:pass@host:5432/nabl_graph
```

`build_checkpointer()` in `app/graph/graph.py` detects the `postgres` prefix and
switches; the SQLAlchemy models already run on Postgres unchanged. If the
Postgres checkpointer package is absent, it logs a warning and falls back to
SQLite rather than failing silently.

Two things worth doing on a real Postgres deployment:

- **Convert the `JSON` columns to `JSONB`** and index them if you start querying
  inside payloads. `JSON` was chosen so one DDL works on both engines; nothing
  in the repository reads them by content.
- **Prune checkpoints on a schedule.** They hold full page text and are the
  largest thing in the database. The `graph_runs` result row is self-contained,
  so a completed run's checkpoint is only needed if you might resume it.

---

## 5. Scaling

| Documents/hour | Shape |
|---|---|
| < 20 | one container, SQLite, defaults |
| 20–200 | 2–4 replicas, Postgres checkpointer, `MAX_REQUESTS_PER_MINUTE` divided across them |
| > 200 | add Celery + Redis; move `_process` out of `BackgroundTasks` into a task queue |

The current design deliberately stops short of Celery — the spec says to add it
only if asynchronous background processing is required, and FastAPI's
`BackgroundTasks` plus a durable checkpoint already survives a restart, because
an interrupted run resumes from its `document_id`. Add the queue when you need
work distributed across machines, not before.

**The rate limiter is per-process.** With N replicas, set
`MAX_REQUESTS_PER_MINUTE` to your provider allowance divided by N, or move to a
shared Redis token bucket.

---

## 6. Operations

**A run is stuck.** `GET /graph/documents/{id}/status` shows `current_node` and
`next_nodes` straight from the checkpoint. Nothing re-executes.

**A run crashed.** `POST /graph/documents/{id}/retry`. Processed chunks are
skipped; only the incomplete work re-runs.

**Provider outage.** `/graph/health?check_llm=true` reports per provider. The
chain fails over automatically and backs off a failing provider; if every
provider is down, chunks fail, recovery exhausts, and the run finishes
`MANUAL_REVIEW_REQUIRED` or `INCOMPLETE` with the failed chunks listed — it
does not hang.

**Rolling back to the legacy pipeline.** Set `GRAPH_PIPELINE_ENABLED=false` and
restart. `app/documents/pipeline.py` falls through to the previous
implementation, which is still present and still tested.

**Cost spike.** Check `metrics.llm_call_count` against page count in
`graph_runs`. A ratio far above `pages / 8` means the retry loop is engaging —
look at `graph_errors` grouped by `error_type`.

---

## 7. Pre-flight checklist

- [ ] At least one provider reachable — `/graph/health?check_llm=true`
- [ ] For Nova: model access granted in the Bedrock console for `NOVA_REGION`,
      and `NOVA_MODEL` is the `us.`-prefixed inference profile
- [ ] `GRAPH_DATABASE_URL` set (or SQLite path on a persistent volume)
- [ ] `graph_storage/` is a durable volume, not container-local
- [ ] `MAX_REQUESTS_PER_MINUTE` divided by replica count
- [ ] `MAX_FILE_SIZE_MB` matches any upstream proxy body limit
- [ ] Checkpoint retention decided (they contain full page text)
- [ ] `pytest tests/graph -q` green against the deployed image
