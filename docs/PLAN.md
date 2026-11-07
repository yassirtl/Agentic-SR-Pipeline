# Execution Plan: Agentic Systematic Review Pipeline

Single source of truth for the project. Each part is built in its own session: read this file, build the part, update the **Progress** table, commit.

## 0. Context

- **Goal:** a multi-agent LangGraph system that takes a research question and runs a PRISMA 2020 systematic review end to end (protocol, search, deduplication, screening, extraction, risk of bias, synthesis, report), transparent and reproducible.
- **Source:** course statement `Project Agentic AI Pipeline.docx` (kept local, not in git).
- **Deliverables:** GitHub code + README (install, config, run, usage) · PRISMA protocol template · end-to-end demo on real topics · technical report (PDF) with individual contribution (solo project) · evaluation (completeness, accuracy, inter-agent consistency, time savings) · video of 3 minutes max.
- **Repo:** https://github.com/yassirtl/Agentic-SR-Pipeline (branch `main`).

## 1. Decisions (fixed)

| Topic | Decision |
|---|---|
| Team | Solo |
| Form | Real Python package (`src/sr_pipeline`) run from the command line on the Mac; no Colab (decided 2026-11-07) |
| LLM | `gpt-4o-mini` now; model configurable **per agent** (cheap model for screening, strong model for extraction/synthesis later) |
| Orchestration | LLM **Coordinator** chooses the next stage after every step; a **code guardrail** enforces PRISMA prerequisites; refusals are logged |
| Sources | OpenAlex + PubMed + arXiv (free); paid sources optional later |
| Full text | Open-access only (PMC, arXiv); others screened on abstract and flagged |
| RAG | FAISS (dense, OpenAI `text-embedding-3-small` embeddings) + BM25 (sparse) hybrid, LLM re-ranker. No torch: PyTorch stops at 2.2.2 on Intel Macs and conflicts with NumPy 2, so local embedding / cross-encoder models are not used |
| Human in the loop | LangGraph `interrupt` at (1) protocol approval, (2) unresolved screening conflicts; `--auto` skips both |
| Synthesis | Narrative always; effect-size table + forest plot only when data are comparable |
| Evaluation | 3 demo topics (chosen later), each with a published review as gold standard; metrics pluggable per field |
| Python | 3.11, `.venv` on the Mac (Intel i7, 16 GB) |

## 2. Conventions

- Commits: author `yassirtl`, short plain messages, **no AI / assistant attribution** anywhere (commits, files, README).
- Notebook answers use the label "Answer:".
- Tests never use the network or the API: scripted fake LLM and recorded API responses. Real runs happen locally on the Mac; the key is in a git-ignored `.env` (`OPENAI_API_KEY=...`), loaded by the config. The user runs real commands (`sr-review ...`) and the outputs in `runs/` are used for the report.
- Every part ends with: tests green, `docs/PLAN.md` progress updated, commit + push, then a plain-language explanation to the user (student tone, only claim what outputs show).
- Pin dependencies (`~=`) so installs stay stable.

## 3. Architecture

### 3.1 State (`state.py`)
`ReviewState` (TypedDict). Lists that several nodes write are append-only: `Annotated[list, operator.add]`.

| Field | Type | Written by |
|---|---|---|
| `question` | str | input |
| `protocol` | dict (PICO, criteria, databases, years) | planner |
| `search_strategy` | dict (query per database) | search_strategy |
| `records` | append list | retrieval |
| `unique_records` | list | retrieval |
| `ta_decisions`, `ft_decisions` | append lists | screening |
| `included` | list of ids | ft_screening |
| `extractions`, `risk_of_bias` | append lists | extraction |
| `synthesis` | dict (text, tables) | synthesis |
| `qa_report` | dict (issues, passed) | critic |
| `prisma_counts` | dict | retrieval, screening, report |
| `completed_stages` | append list (order kept) | every stage |
| `audit_log` | append list of dicts | every node |
| `next_stage`, `steps_taken` | str, int | coordinator |

### 3.2 Stages and guardrail (`guardrail.py`)
Stages in PRISMA order: `planner, search_strategy, retrieval, ta_screening, ft_screening, extraction, synthesis, critic, report`.

```
PREREQUISITES = {
  planner: [], search_strategy: [planner], retrieval: [search_strategy],
  ta_screening: [retrieval], ft_screening: [ta_screening], extraction: [ft_screening],
  synthesis: [extraction], critic: [synthesis], report: [critic],
}
```
Rules:
1. A stage is **fresh** if it has run and its last run came after the last run of each prerequisite, which must itself be fresh (so re-running retrieval makes screening stale).
2. A proposed move is allowed only if all its prerequisites are fresh. `report` therefore needs the whole chain fresh.
3. Illegal or unknown move: **deterministic fallback** to the earliest stage (PRISMA order) that is not fresh. Never re-ask the LLM. Log `{proposed, reason, substituted}`.
4. The run ends only after `report` (report → END).
5. **Step budget** `MAX_STEPS = 40`: when exhausted, end with an audit entry "budget exhausted".

### 3.3 Coordinator (`coordinator.py`)
- Sees `summarize_state(state)` only (counts, flags, completed stages, last critic issues), never raw record lists.
- Returns a Pydantic `Decision(next: Literal[stages], rationale: str)`.
- Its proposal goes through the guardrail; the audit log stores proposal, rationale, and outcome.

### 3.4 LLM access (`llm.py`)
- One helper: `structured_call(schema, prompt, agent=...) -> schema instance`.
- Real implementation: `ChatOpenAI(model=cfg.model_for(agent), temperature=0).with_structured_output(schema)`.
- Tests inject a scripted fake (LangChain fake chat models do **not** support `with_structured_output`; verified).
- Disk cache keyed on (model, schema, prompt) under `.cache/` → re-runs are identical and free. Token usage + cost recorded in the audit log.

### 3.5 Graph (`graph.py`)
- `START → coordinator`; `coordinator →` conditional edge on `next_stage` to a stage or `END`; every stage → `coordinator`; `report → END`.
- Compile with a checkpointer (`MemorySaver` in tests, SQLite for real runs); invoke with `{"configurable": {"thread_id": run_id}, "recursion_limit": 100}` (default 25 is too low for supervisor loops).

### 3.6 Layout
```
pyproject.toml   README.md   docs/PLAN.md
protocols/template.yaml
src/sr_pipeline/  config.py state.py guardrail.py coordinator.py llm.py graph.py audit.py cli.py
                  agents/  sources/  rag/  reporting/
evaluation/  tests/
runs/<run_id>/   (git-ignored: state, audit.jsonl, outputs)
```

## 4. Parts

Each part: **Goal · Build · Done when · User runs · Explain**.

### Part 2: Skeleton
- **Build:** `state.py`, `guardrail.py`, `llm.py` (interface + fake), `coordinator.py`, `graph.py` with 9 placeholder stage nodes; `tests/test_skeleton.py`.
- **Done when:** `pytest` passes these tests:
  (a) happy path: completed stages in PRISMA order, ends after report;
  (b) **cheating coordinator** proposes `extraction` right after `planner` → guardrail runs `search_strategy`, refusal in audit log;
  (c) critic loop: after critic, coordinator sends back to synthesis then proposes `report` → refused (critic stale) → critic runs again → report;
  (d) step budget ends the run.
- **User runs:** `.venv/bin/pip install -e ".[dev]"`, `.venv/bin/pytest -v` (seconds, $0).
- **Explain:** state reducers, the guardrail table and freshness rule, supervisor loop, checkpointer and thread id.

### Part 2b: Skeleton fixes (from the Part 2 review)
- **Fix 1: LLM cache + cost (§3.4).** In `llm.py`, wrap `OpenAILLM.structured_call` with a disk cache under `.cache/llm/` keyed on sha256 of (model, schema name + JSON schema, prompt); a cache hit makes no API call. Capture token usage (`with_structured_output(..., include_raw=True)` → `usage_metadata`) and compute cost from a per-model price table (input/output per 1M tokens, editable). Expose usage per call so nodes can add `{model, input_tokens, output_tokens, cost_usd, cached}` to their audit entry. Build the ChatOpenAI client once per model, not per call.
- **Fix 2: no duplicate decisions after a loop-back.** Every decision/extraction item carries `paper_id` and `round` (the stage's run count, from `completed_stages`). Add a helper `latest(items, key="paper_id")` that keeps only the most recent round per paper; later parts (6, 7, 8) must read decisions through it. The full history stays in the state for the audit trail.
- **Fix 3: Coordinator never crashes.** If `structured_call` raises (bad JSON, validation error, API error after retries), fall back to `earliest_stale(completed)` and log `{outcome: "llm_error", error, substituted}`; the step still counts toward the budget.
- **Done when:** new tests pass: cache hit returns the same object without calling the model (fake chat model / monkeypatch, no network); cost computed from fake usage; `latest()` keeps the last round per paper; a coordinator whose LLM raises still completes the review with `llm_error` entries. All Part 2 tests still pass.
- **Explain:** why caching is the real reproducibility guarantee, how cost is measured, why history is kept but only the latest decision is used.

### Part 3: Config, protocol template, Planner
- **Build:** `config.py` (YAML + env; per-agent model; `max_records`; paths), `protocols/template.yaml` (PICO, inclusion/exclusion, databases, years, languages, study designs), `agents/planner.py` (question → protocol), protocol approval `interrupt` (skipped with `auto=True`), `audit.py` (JSONL writer in `runs/<run_id>/`).
- **From the 2b review:** the LLM cache dir (`.cache/llm`) is relative to the working directory; make it a config path (absolute, project root) so Colab and local runs find the same cache.
- **Done when:** fake-LLM test produces a valid protocol; interrupt pauses and resumes with edits.
- **User runs:** planner on one question, < 1 min, < $0.01 (skipped at build time; first real call happens in Part 4).
- **Explain:** PICO, why the protocol is fixed first, interrupt/resume.

### Part 4: Search strategy + source connectors
- **Build:** `agents/search_strategy.py` (Boolean query per database, MeSH for PubMed), `sources/openalex.py`, `sources/pubmed.py` (E-utilities, ≤3 req/s without key), `sources/arxiv.py`; common `Record` model (id, doi, title, abstract, authors, year, venue, source, oa_pdf_url); HTTP caching; retries; `max_records`.
- **Done when:** unit tests with recorded API responses (no network in tests); live smoke search returns records from all 3 sources.
- **User runs:** live search for one topic, 1–3 min, $0.
- **Explain:** Boolean logic, MeSH, why several databases, rate limits.

### Part 5: Retrieval, deduplication, full text, RAG
- **Build:** `agents/retrieval.py`; dedup (DOI exact, then normalized-title fuzzy match) with counts for PRISMA; PDF download for open-access + text extraction; chunking; `rag/` hybrid index (FAISS over OpenAI embeddings + BM25 via `rank-bm25`, reciprocal rank fusion), LLM re-ranker (gpt-4o-mini scores the top ~20 passages); embeddings cached on disk; index saved per run.
- **Done when:** dedup tests on crafted duplicates; retrieval test returns the right chunk for a known query.
- **User runs:** index ~150 papers, 5–15 min, ~$0.05 (embeddings via API; cached so re-runs are free).
- **Explain:** dense vs sparse search, fusion, re-ranking, why chunk.
- **Coordinator rule example:** too few records → may send back to search_strategy.

### Part 6: Screening (title/abstract + full text)
- **Build:** `agents/screening.py`; screener A (strict prompt) and B (lenient prompt), structured `ScreeningDecision(decision, criterion, reason, quote)`; adjudicator on disagreement; unresolved → `interrupt` (or auto-include in `--auto`); full-text round uses RAG passages per criterion; Cohen's kappa computed; PRISMA counts updated; concurrency + cache.
- **From the 2b review:** (a) when screening runs concurrently, write cache files atomically (temp file + rename) so a half-written file never breaks a read; (b) after a retrieval re-run, filter `latest()` results to papers still in `unique_records`, so a paper dropped by the new search does not keep an old decision.
- **Done when:** fake-LLM tests for agree / disagree / adjudicate / human paths; kappa unit test.
- **User runs:** ~50 abstracts first (~2 min, ~$0.05), then full topic (10–20 min, ~$0.50–1).
- **Explain:** dual screening, adjudication, kappa, recall over precision at screening.

### Part 7: Extraction + risk of bias
- **Build:** `agents/extraction.py`; fields derived from protocol (PICO + design, sample size, outcome, effect); every field with source quote via RAG; risk-of-bias tool chosen by design (RoB 2 trials, ROBINS-I non-randomized, simple checklist otherwise), domain-level judgments with reasons.
- **Done when:** fake-LLM tests; schema validation; missing values stay "not reported" (never invented).
- **Explain:** evidence tables, risk-of-bias domains.

### Part 8: Synthesis + Critic
- **Build:** `agents/synthesis.py` (narrative grouped by outcome/design; effect-size table + random-effects pooling and forest plot only when ≥3 comparable effects); `agents/critic.py` (protocol adherence, unsupported claims, count consistency, missing studies) → `qa_report`; coordinator may loop back.
- **Done when:** tests for the critic loop and the "not comparable → narrative only" path.
- **Explain:** narrative vs meta-analysis, why a critic.

### Part 9: Reporting, exports, CLI
- **Build:** `reporting/` PRISMA 2020 flow diagram (matplotlib, from counts), manuscript Markdown + DOCX, evidence tables CSV, BibTeX, run summary (time, tokens, cost); `cli.py`: `sr-review "question" [--auto] [--config] [--resume RUN_ID]`.
- **Done when:** full fake-LLM end-to-end run produces every output file.
- **User runs:** first full real review, 30–60 min, ~$1–3.

### Part 10: Evaluation, documentation, video
- **Build:** `evaluation/` (recall/precision vs published review's included list, accuracy vs user-labelled sample, kappa, time/cost vs manual estimate; pluggable metrics); README (install, config, run, usage, results); technical report PDF incl. individual contribution; 3 topics chosen with the user.
- **User does:** pick 3 topics + gold lists (1–2 h), label ~30 abstracts (~1 h), 3 full runs (~3 h, ~$5–15 total), record video (1–2 h).

## 5. Estimates

- Build: ~10–14 h of session time + ~6–8 h user reading. ~1–2 parts per 5-hour usage window → about 1–2 weeks of calendar time.
- API (gpt-4o-mini): ~$1–3 per full review; ~$5–15 for the whole project. Re-runs from cache ≈ $0. Real cost will be measured from the audit log.

## 6. Progress

| Part | Status | Notes |
|---|---|---|
| 1 Architecture + strategy deck | done | 2026-11-07 |
| 2 Skeleton | done | 2026-11-07 · 7 tests green; LLM disk cache + token cost (§3.4) done in 2b |
| 2b Skeleton fixes (cache + cost, latest decision, coordinator fallback) | done | 2026-11-07 · 15 tests green; usage via `call_with_usage()`; stages must tag items with `stage_round()` and read via `latest()` |
| 3 Config + Planner | done | 2026-11-07 · 27 tests green; `config.yaml` + env (`SR_*`), paths absolute under project root (cache fix from 2b); `protocol.py` (LLM drafts PICO/criteria, template fixes databases/years/languages); approval is a separate non-stage node (`build_graph(approve_protocol=True)`), resume with `True` or `{"edits": {...}}`; `AuditWriter.sync()` writes only new entries; Colab run skipped (tests only) |
| 4 Search + sources | done | 2026-11-07 · 44 tests green; `.env` loaded by `load_config` (shell wins), Colab wording removed, `contact_email` + `http_cache_dir` in config, keys `NCBI_API_KEY`/`OPENALEX_API_KEY` from env only; LLM gives concepts (keywords + MeSH), code builds each database's Boolean query (`Source.build_query`); `sources/http.py` = disk cache (keys excluded), retries (429/5xx, Retry-After), per-host rate limit; tests replay real recorded exchanges (`tests/fixtures/sources/`, incl. a real OpenAlex 429). Found live: OpenAlex `search=` hits full text (use `title_and_abstract.search` filter), cursor paging needs a key (page paging used), no key = $0.10/day shared budget per IP; PubMed needs `sort=relevance` when capped. Live connector check OK (all 3 sources). Planner+search smoke (`python -m sr_pipeline.search_smoke "question"`) left for the user |
| 5 Retrieval + RAG | next | retrieval node calls `sources.search_all(state['search_strategy'], ...)`; log `total_hits`, `capped`, `searched_at` per database for PRISMA; coordinator can see capped / too few hits |
| 6 Screening | todo | |
| 7 Extraction + RoB | todo | |
| 8 Synthesis + Critic | todo | |
| 9 Reporting + CLI | todo | |
| 10 Evaluation + docs + video | todo | |

## 7. Starting a part in a new session

Paste:
> Read docs/PLAN.md. Build Part N only, following the plan. Explain it to me step by step as you go, then update the Progress table and commit.
