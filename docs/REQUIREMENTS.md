# Requirements checklist (course brief → this project)

Every requirement of the course brief ("Development of an Agentic AI Pipeline for Conducting Rigorous, Transparent, and Reproducible Systematic Reviews on Any Given Topic"), with where it is met. ✅ met · ⚠️ partly met · ❌ not met. Paths are under `src/sr_pipeline/` unless they start with a folder of the repository.

## 1–2. Overview and objectives

| # | Requirement (brief) | Status | Where / evidence |
|---|---|---|---|
| 1 | Autonomous multi-agent system for end-to-end systematic reviews | ✅ | 9 agents in one LangGraph graph (`graph.py`, `pipeline.py`); a full run goes from a question to a report (`cli.py`, `sr-review`) |
| 2 | LangChain for agent orchestration and workflow management | ✅ | LangGraph `StateGraph` (`graph.py`); LLM calls through `langchain-openai` (`llm.py`) |
| 3 | LangChain for **tool use** | ❌ | The agents call the LLM for structured answers (`with_structured_output`, `llm.py`); the tools (search APIs, PDF reader, counts) are called by code, never chosen by the LLM. **Gap A** |
| 4 | RAG for grounded, evidence-based reasoning | ✅ | `rag/`: chunking, embeddings, FAISS + BM25 hybrid index, LLM re-ranking; passages feed full-text screening and extraction (`agents/screening.py`, `agents/extraction.py`) |
| 5 | Question or topic in, full PRISMA review out | ✅ | `sr-review "question"` or `--topic`; app "Start a new review" |
| 6 | Sources: PubMed, Semantic Scholar, arXiv, OpenAlex, etc. | ⚠️ | OpenAlex, PubMed, arXiv (`sources/`); **Semantic Scholar not connected** (listed as an example in the brief). **Gap F** |
| 7 | PRISMA workflow: protocol, search strategy, dual screening, extraction, risk of bias, qualitative/quantitative synthesis, report | ✅ | `agents/planner.py`, `search_strategy.py`, `screening.py` (screeners A, B + adjudicator), `extraction.py` (+ RoB), `synthesis.py` (random-effects meta-analysis when ≥ 3 studies, SWiM otherwise), `reporting/` |
| 8 | Transparency: logging, citation tracking, decision rationales, reproducible traces | ✅ | `audit.jsonl` per run (`audit.py`), a verified quote behind every exclusion and extracted value, S-labels and BibTeX, disk cache of every LLM / database answer (`llm.py`, `sources/http.py`) |
| 9 | Human in the loop at critical points | ✅ | LangGraph `interrupt`: protocol approval, unresolved screening conflicts, budget cap (`graph.py`, `coordinator.py`); answered in the CLI or the app |
| 10 | General-purpose: any topic, minimal configuration | ⚠️ | A free question needs no configuration (CLI and app; tested end to end with the scripted LLM). **All real runs so far used the evaluation topics.** **Gap E** |

## 3. Scope

| # | Requirement (brief) | Status | Where / evidence |
|---|---|---|---|
| 11 | Question intake and protocol generation | ✅ | `agents/planner.py`, `protocols/template.yaml` |
| 12 | Search strategies: Boolean, MeSH | ✅ | `agents/search_strategy.py`; MeSH `[mh]` terms and design filters in `sources/pubmed.py`, `sources/design_filters.py` |
| 13 | Multi-source retrieval and deduplication | ✅ | `agents/retrieval.py`, `dedup.py` (DOI, PMID, PMCID, fuzzy title) |
| 14 | Title/abstract and full-text screening with rationale | ✅ | `agents/screening.py`: verdict per criterion with a verified quote; `tables/screening.csv` |
| 15 | Structured extraction and evidence tables | ✅ | `agents/extraction.py`; `reporting/tables.py` (`tables/*.csv`), study table in the article |
| 16 | Quality / risk-of-bias assessment | ✅ | RoB 2, ROBINS-I, QUADAS-2, checklist or the topic's own instrument (`agents/extraction.py`) |
| 17 | Narrative synthesis and meta-analysis preparation | ✅ | `agents/synthesis.py`, `meta.py`, forest plots (`forest.py`) |
| 18 | PRISMA flow diagram data | ✅ | `reporting/prisma.py` (diagram + arithmetic checks), `prisma_counts.json` |
| 19 | Evidence maps | ✅ | `reporting/findings.py` (model × device map, PNG + table) |
| 20 | Final review manuscript draft | ✅ | `reporting/manuscript.py` → `manuscript.md` / `.docx` (PRISMA 2020 order, evidence gaps, research agenda) |
| 21 | Persistent memory, versioning, audit trails | ✅ | LangGraph SQLite checkpoint (resume, `cli.py`), items tagged by round (`state.py`), `audit.jsonl` |

## 4. Technical approach

| # | Requirement (brief) | Status | Where / evidence |
|---|---|---|---|
| 22 | Coordinator / Planner agent decomposes and manages the workflow | ✅ | `coordinator.py` (LLM picks the next stage) + `guardrail.py` (code enforces PRISMA order); `agents/planner.py` |
| 23 | Search Strategy agent designs and refines queries | ✅ | `agents/search_strategy.py` (concepts → Boolean query per database, refined on hit counts) |
| 24 | Retrieval agent executes searches **and manages RAG index construction/updates** | ⚠️ | Searches and dedup in `agents/retrieval.py`; the RAG index is built in the screening and extraction stages (`agents/screening.py`, `agents/extraction.py`), because it needs the full texts those stages fetch. **Gap G** (document or move) |
| 25 | Screening agents (T/A + full text) with reasoning | ✅ | `agents/screening.py` |
| 26 | Extraction & appraisal agent | ✅ | `agents/extraction.py` |
| 27 | Synthesis & writing agent: narrative, tables, final report | ✅ | `agents/synthesis.py`, `reporting/` (writer: `reporting/writer.py`) |
| 28 | Critic / QA agent: consistency, bias, completeness, protocol adherence | ✅ | `agents/critic.py`: count consistency, every outcome covered, design vs protocol, unsupported claims (numbers, risk of bias), loop back to synthesis |
| 29 | LangChain / LangGraph: agent definition, **tool calling**, memory, routing, state | ⚠️ | Agents, memory (checkpoint), routing (coordinator + conditional edges), state (`state.py`) ✅; **tool calling ❌** (Gap A) |
| 30 | RAG: hybrid (dense + sparse), re-ranking, **context compression**, vector store | ⚠️ | FAISS (dense) + BM25 (sparse) fused by reciprocal rank fusion (`rag/index.py`), LLM re-ranking (`rag/rerank.py`), only the top passages per question go to the LLM; **no separate compression step**. **Gap D** |
| 31 | LLM backbone: high-capability, long context | ⚠️ | Model configurable per agent (`config.py`); runs used gpt-4o-mini (128k context) for cost; stronger models tested for the adjudicator. **Gap H** (document the choice) |
| 32 | Tools: search APIs, PDF parsing, citation managers, **PRISMA checklist validators**, export (Markdown, DOCX, BibTeX) | ⚠️ | APIs (`sources/`), PDF and PMC XML (`sources/fulltext.py`), BibTeX, Markdown, DOCX (`reporting/`); **PRISMA checking covers the flow-diagram arithmetic only, not the 27-item PRISMA 2020 checklist**. **Gap C** |
| 33 | Observability: tracing of decisions, retrieved documents, intermediate artifacts | ✅ | `audit.jsonl` (every node, with times and costs), `state.json` (decisions with passages and quotes), the app's run detail page |

## 5–6. Deliverables and success criteria

| # | Requirement (brief) | Status | Where / evidence |
|---|---|---|---|
| 34 | Source code + documentation on GitHub | ✅ | https://github.com/yassirtl/Agentic-SR-Pipeline |
| 35 | Configurable protocol template aligned with PRISMA | ✅ | `protocols/template.yaml`, `evaluation/topics.yaml` |
| 36 | End-to-end demonstration on at least one real topic | ✅ | Real runs on T1, T2, T3, D0 (`evaluation/results/`); free demo replay in the app (`docs/demo/`) |
| 37 | Technical report and user guide | ✅ | `docs/report/report.pdf`, `README.md` |
| 38 | Evaluation: completeness, accuracy, inter-agent consistency, time savings | ⚠️ | Completeness (gold funnel), consistency (kappa, coordinator refusals), time (PredicTER) ✅; accuracy: precision lower bound + a per-stage check that is **not blind**. **Gap I** (blind labels, user) |
| 39 | Full pipeline with minimal human intervention | ✅ | `--auto` runs end to end; two optional pauses |
| 40 | Methodologically sound, fully referenced, transparent; traceable rationales | ✅ | Verified quotes, included-study references and BibTeX, PRISMA diagram, audit |
| 41 | Significant time reduction, rigor kept | ✅ | Report §4.6: under 30 min and about $1 per run vs 69–80 person-days for the same work (with the stated caveats) |

## Submission requirements

| # | Requirement (brief) | Status | Where / evidence |
|---|---|---|---|
| 42 | All source code on GitHub or GitLab | ✅ | GitHub, branch `main` |
| 43 | README documenting **all the steps followed during development** and setup (install, configuration, execution, usage) | ⚠️ | Install, configuration, run, usage ✅; **development steps only in `docs/PLAN.md`, not in the README**. **Gap B** |
| 44 | Video, 3 minutes max | ❌ | Script ready (`docs/video.md`); **to record** (user) |
| 45 | Individual contribution in the final report | ⚠️ | Report §7 drafted; **the author checks it and writes it in their own words** (user) |

## Gaps and fixes

| Gap | What is missing | Fix | Cost | Who |
|---|---|---|---|---|
| A | LLM tool calling | One agent made a real tool-calling agent (the LLM chooses which tool to call and when) | code, $0 to test; one real run to show it | build |
| B | Development steps in the README | Section "How it was built" (the parts of `docs/PLAN.md`, in short) | $0 | build |
| C | PRISMA 2020 checklist validator | Code that checks the article against the 27 PRISMA 2020 items and prints the checklist (item, reported or not, section) | $0 | build |
| D | Context compression | An explicit compression step: from each retrieved passage, keep only the sentences relevant to the question before the LLM reads it | $0 code; changes new runs only | build |
| E | A real run on a free question | One real run from the app on a question that is not an evaluation topic (100 papers) | about $0.50 | user starts it |
| F | Semantic Scholar | A fourth source connector, or state it is not used (the brief lists it as an example) | code, free API | decide |
| G | RAG index in the retrieval agent | State in the README/report where the index is built and why | $0 | build |
| H | Model choice | State in the README/report why gpt-4o-mini and how to switch per agent | $0 | build |
| I | Blind accuracy | Fill the 30 blind labels in the app | about 1 h | user |
| — | Video, contribution text | Record; check §7 | — | user |
