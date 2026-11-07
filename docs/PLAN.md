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

### Part 4b: Search precision fixes (from the first real run)
First real run (2026-11-07, `runs/20261107-145953-7b76f2`, question "Does exercise reduce depressive symptoms in older adults?", cost $0.0004): the query was population AND intervention only, so it matched 349,263 (OpenAlex) and 111,813 (PubMed) records; the 500 kept per database were mostly off-topic. Records mentioning depression: OpenAlex 28/500, PubMed 58/500, arXiv 9/500. Leaving the outcome out of the search maximizes recall only when *all* hits are screened; with a `max_records` cap it throws recall away instead.
- **Fix 1: outcome block when it defines the topic.** Search strategy prompt: add the outcome as a third block when it is specific (e.g. depression) or when P AND I is far too broad; a design block (e.g. randomized trials; PubMed `randomized controlled trial[pt]` + keywords) only if the protocol restricts designs.
- **Fix 2: count-and-refine loop inside the search_strategy stage.** After building queries, fetch hit counts only (OpenAlex `meta.count`, PubMed `esearch` count, arXiv `totalResults`; cached, no records downloaded). If a database returns more than `broad_factor x max_records` (config, default 5), or fewer than `min_hits` (default 20), give the counts back to the LLM and ask for a narrower/broader plan; at most 2 refinements. Store every attempt (concepts, queries, counts, reason) in `search_strategy.attempts` for the report (PRISMA-S transparency).
- **Fix 3: Planner stays on the question.** Outcomes come from the question (primary outcome first); extra outcomes only from user hints. Comparator: when the question asks about an effect, write a control (no intervention, usual care, any control) rather than "none".
- **Fix 4: database choice.** Template `databases: auto` (new default): the Planner picks from openalex/pubmed/arxiv with a one-line reason (PubMed for health, arXiv for CS/physics/quantitative fields, OpenAlex always). An explicit list in the template still wins. The human sees and can edit the choice at protocol approval.
- **Done when:** tests for the refine loop (too broad → narrower, too narrow → broader, stops after 2), outcome block present for a specific outcome, `auto` database choice; all earlier tests pass. Re-run the smoke on the same question and report the new hit counts and the share of records mentioning depression (before: 6%, 12%, 2%), so the report has a before/after.
- **User runs:** `python -m sr_pipeline.search_smoke "Does exercise reduce depressive symptoms in older adults?"` (< 2 min, < $0.01).

### Part 4c: Search recall fixes (from the Part 4b run)
Run `runs/20261107-151313-357c01` ($0.0012): queries are now on topic, but three things lose eligible studies.
- **Fix 1: no silent truncation.** OpenAlex 3,769 and PubMed 2,375 hits were "ok" (≤ `broad_factor` × `max_records` = 2,500... and OpenAlex never got under it), yet only 500 per database were downloaded: up to 87% of matching records never reach screening and are not reported. Change the rule: `max_records` (default 3,000 per database) is the download ceiling and the "ok" range is `min_hits` ≤ hits ≤ `max_records`; drop `broad_factor`. If the ceiling is still exceeded after refinements, download the top `max_records`, set `capped`, and record "not retrieved: N" so the PRISMA diagram and the report state it. Screening ~5,000 abstracts costs ~$1–2 with gpt-4o-mini, acceptable.
- **Fix 2: the Planner must not invent design restrictions.** The question did not restrict designs, but the Planner set `study_designs = [RCT, cohort, case-control]`, which triggered a design block. Designs come from the question or template hints only; otherwise `study_designs = []` and no design block.
- **Fix 3: proper design filters when designs are restricted.** The design block used only two of the three listed designs, as exact phrases (`"randomized controlled trial"[tiab]` misses "randomised", "RCT", "randomized trial"), plus `"Cohort Studies"[pt]`, which is not a publication type and matches nothing. Use validated filters written in code, not by the LLM: for RCTs, the Cochrane highly sensitive filter (PubMed: `randomized controlled trial[pt] OR controlled clinical trial[pt] OR randomized[tiab] OR placebo[tiab] OR randomly[tiab] OR trial[ti]`, keyword equivalent for OpenAlex/arXiv); cohort/case-control as MeSH `[mh]` + keyword variants. Every listed design must appear.
- **Done when:** tests for each fix; re-run the same question and report hits, downloaded, not retrieved, per database.
- **User runs:** `python -m sr_pipeline.search_smoke "Does exercise reduce depressive symptoms in older adults?" --check-word depress` (< 3 min, < $0.01).

### Part 4d: Design suggestion + real narrowing (from the Part 4c run)
Run `runs/20261107-152601-4f1ef5`: with no design restriction the question matches ~29,600 (OpenAlex) and ~10,200 (PubMed) records, and the refine loop added synonyms instead of narrowing (29,587 → 29,610 → 29,621). The query holds noisy terms: `aged` (matches "aged 18–65" in any adult study), `training`, `fitness`, `aerobic`, `sadness`.
- **Fix 1: Planner suggests, the human decides.** New protocol field `suggested_designs` + `design_reason` (e.g. effect questions such as "does X reduce Y" → randomized controlled trials). It is NOT applied by itself: `study_designs` stays as the 4c guard sets it. At protocol approval the suggestion is shown and the human accepts it (`{"edits": {"study_designs": [...]}}` or a shortcut `{"accept_suggested_designs": true}`). In `--auto` it is never applied; the audit log records "suggested, not applied". Design restriction must match the gold-standard review in Part 10 evaluation.
- **Fix 2: narrowing that keeps recall.** Rewrite the refinement prompt: when too broad, the allowed moves are (a) remove ambiguous or generic terms (`training`, `fitness`, `sadness`), (b) turn a short ambiguous word into a phrase (`aged` → `"aged 65"`, `"aged 60"`), (c) add the outcome block if missing. Not allowed: removing a specific synonym eligible studies use (`elderly`, `depressive symptoms`). The LLM returns `removed_terms` and `changed_terms` with a one-line reason each, stored in `search_strategy.attempts`. Code check: a refinement whose counts change by less than 5% versus the previous attempt is flagged `no_progress` and the loop stops early.
- **Fix 3: safety net unchanged.** Still above `max_records` after refinements → top `max_records` downloaded, rest reported as not retrieved (4c).
- **Model:** keep gpt-4o-mini; if narrowing still fails, set `models.search_strategy` to a stronger model in `config.yaml` (no code change).
- **Done when:** tests for suggestion not applied by default / applied on acceptance / never in auto; refinement output schema with removed/changed terms; `no_progress` stop. Live re-run of the same question twice: (1) no design restriction, (2) suggestion accepted (RCTs); report counts per attempt, removed terms, downloaded, not retrieved.
- **User runs:** `python -m sr_pipeline.search_smoke "Does exercise reduce depressive symptoms in older adults?" --check-word depress` (and with the suggestion accepted), < 5 min, < $0.02.

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
| 4b Search precision fixes | done | 2026-11-07 · 63 tests green. Planner: outcomes = the question's construct only (no "reduction in", no extra outcomes), population as stated, comparator = a control; `databases: auto` (default) → Planner picks + one-line reason, code always keeps OpenAlex, explicit list wins, editable at approval. Search: outcome block when specific, design block only if designs restricted (PubMed `[pt]` via `Concept.publication_types`), `Concept.role`; `Source.count()` (same filters as `search()`) + `count_all`; refine loop (`broad_factor` 5, `min_hits` 20, `max_refinements` 2 in config/env), every attempt in `search_strategy.attempts` (concepts, queries, counts, status, reason, `same_as`, `overlap`), final = fewest databases out of range (ties: latest); one audit entry with summed tokens/cost. Live re-run (`runs/20261107-151313-357c01`, $0.0012, `--check-word depress`): OpenAlex 349,263 → 3,769 hits, depression share 6% → 92%; PubMed 111,813 → 2,375, 12% → 97%; arXiv 2% → not searched (Planner choice). Caveats: the share is partly circular (the word is now in the query), so it measures staying on topic, not recall (recall is checked in Part 10 against a gold list); OpenAlex stayed above 2,500 after 2 refinements that changed little (gpt-4o-mini); LLM wrote `"Cohort Studies"[pt]` (a MeSH heading, not a publication type: matches nothing, harmless). An intermediate run (`runs/20261107-151017-2f5bb8`) showed Planner drift (population narrowed to "with depressive symptoms", outcomes as result phrases, depression terms in the population block, an identical refinement): fixed by the stricter prompts + `same_as`/`overlap` diagnostics. If drift returns, use `models.planner` (stronger model) |
| 4c Search recall fixes | done | 2026-11-07 · 75 tests green. Fix 1: `broad_factor` dropped; `max_records` = download ceiling (default 3,000, max 10,000 = OpenAlex/PubMed paging limit) and top of the "ok" range; still above after refinements → top `max_records` downloaded, `search_strategy.capped` / `not_retrieved` per database (expected, from counts) + `SearchResult.not_retrieved` (actual, in `search_results.json`). Fix 2: Planner prompt + schema rule, and code guard `choose_designs`: template designs win exactly, else LLM designs kept only if the question names a design word, else `[]`; audit `study_designs_source` / `study_designs_dropped`. Fix 3: `sources/design_filters.py`: one code-built design block, every listed design OR-ed (RCT = Cochrane HSSS sens.+prec. version + randomised/RCT; cohort, case-control, cross-sectional = `[mh]` + keywords; others searched by name, flagged `unvalidated`); `Concept.title_terms` (PubMed `[ti]`, arXiv `ti:`, OpenAlex title+abstract); LLM design blocks dropped (`llm_design_blocks_dropped`), counts include the design block. Live re-run (`runs/20261107-152601-4f1ef5`, $0.0012, 101 s): no designs (prompt alone gave `[]`, guard not triggered), so no design block. OpenAlex 29,621 hits / 3,000 downloaded / 26,621 not retrieved; PubMed 10,188 / 3,000 / 7,188; depress share 96% / 98%. The 4b counts (3,769 / 2,375) were low only because of the invented design block; with `max_records` 500, 4b silently dropped 87% / 79%. **Open issue:** told "too broad", gpt-4o-mini *added* synonyms in both refinements (29,587 → 29,610 → 29,621), so refinement does not narrow. To decide with the user before Part 5: accept the cap (reported), restrict designs in the template (e.g. RCTs only), raise `max_records`, or a stronger `models.search_strategy` / a narrowing rule |
| 4d Design suggestion + real narrowing | done | 2026-11-07 · 91 tests green. Fix 1: `ProtocolDraft.suggested_designs` + `design_reason` (defaults empty); `study_designs` still gated by `choose_designs`; approval accepts `{"accept_suggested_designs": true}` (combinable with `edits`, an explicit `study_designs` edit wins, empty suggestion → asked again) via `planner.apply_answer` (shared with the smoke); audit `design_suggestion` = `suggested, not applied` / `accepted by human` / `none suggested`; never applied in `--auto`; the search prompt never sees the suggestion. Fix 2: refinement calls return `RefinedPlan` (`removed_terms`, `changed_terms` with reasons); refine prompt allows only (a) drop generic terms, (b) short word → phrase, (c) add outcome block; no adding keywords, no dropping specific synonyms, MeSH kept; code records the real `term_diff` (removed/added) per attempt; `no_progress` (every database's count moved < 5% vs previous attempt) stops the loop, and a `no_progress` attempt never wins a tie in `choose_attempt`. Also fixed: connectors set `capped` only when hits > `max_records` (was: any shortfall; a few records a database counts but does not return, e.g. OpenAlex works without a title, stay in `not_retrieved`). Updated 3 older tests that scripted identical plans (now a no-progress stop). Live runs, same question, $0.0020 total: (1) no design restriction (`runs/20261107-155752-91a065`, cached re-run `runs/20261107-155946-feebb9`): attempt 1 OpenAlex 30,087 / PubMed 10,234 → attempt 2 6,870 / 6,215 (removed `aged`→`aged 65`, `training`, `fitness`, `sadness`, `workout`, `65 years`, `senior citizens`) → attempt 3 6,867 / 6,215 = no progress, stop; attempt 2 used: 3,000 downloaded each, not retrieved 3,870 / 3,215; depress share 94% / 98%. (2) RCT suggestion accepted (`runs/20261107-160030-c56f7f`, cached re-run `runs/20261107-160209-aec820`): attempt 1 8,117 / 2,760 → attempt 2 1,902 / 1,504, all in range; downloaded 1,901 / 1,497, not retrieved 1 / 7 (not returned by the database, not the cap); depress share 89% / 98%. Observations: the LLM's own `removed_terms` list is incomplete (it removed 7 terms, listed 2 + 1 change), so `term_diff` is the reliable record; in run 1 attempt 3 it dropped `dysthymia` (a forbidden move), harmless now that no-progress attempts lose ties. Planner comparator came out as "no intervention" (not "any control"): to watch. Without a design restriction ~6,500 records per database stay beyond the 3,000 ceiling: before Part 5 the user chooses between the RCT restriction (matches most effect reviews; must match the Part 10 gold standard) and a higher `max_records` |
| 5 Retrieval + RAG | next | retrieval node calls `sources.search_all(state['search_strategy'], ...)` (pass `max_records`, same as the strategy's `targets`); log `total_hits`, `capped`, `not_retrieved`, `searched_at` per database for PRISMA (`not_retrieved` goes into `prisma_counts` and the flow diagram); coordinator can see capped / too few hits |
| 6 Screening | todo | |
| 7 Extraction + RoB | todo | |
| 8 Synthesis + Critic | todo | |
| 9 Reporting + CLI | todo | |
| 10 Evaluation + docs + video | todo | |

## 7. Starting a part in a new session

Paste:
> Read docs/PLAN.md. Build Part N only, following the plan. Explain it to me step by step as you go, then update the Progress table and commit.
