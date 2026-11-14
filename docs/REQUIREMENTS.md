# Requirements checklist (course brief → this project)

Every requirement of the course brief ("Development of an Agentic AI Pipeline for Conducting Rigorous, Transparent, and Reproducible Systematic Reviews on Any Given Topic"), with where it is met. ✅ met · ⚠️ partly met · ❌ not met. Paths are under `src/sr_pipeline/` unless they start with a folder of the repository.

## 1–2. Overview and objectives

| # | Requirement (brief) | Status | Where / evidence |
|---|---|---|---|
| 1 | Autonomous multi-agent system for end-to-end systematic reviews | ✅ | 9 agents in one LangGraph graph (`graph.py`, `pipeline.py`); a full run goes from a question to a report (`cli.py`, `sr-review`) |
| 2 | LangChain for agent orchestration and workflow management | ✅ | LangGraph `StateGraph` (`graph.py`); LLM calls through `langchain-openai` (`llm.py`) |
| 3 | LangChain for **tool use** | ✅ | The critic is a LangChain tool-calling agent (`bind_tools`, `llm.py` `call_with_tools`, `agents/critic.py`): it chooses when to call `lookup_study`, `outcome_summary`, `check_quote`; calls logged in the audit (tested on a real call: 105 tool calls on T2). Other agents use structured output; their tools (search APIs, PDF reader) are called by code |
| 4 | RAG for grounded, evidence-based reasoning | ✅ | `rag/`: chunking, embeddings, FAISS + BM25 hybrid index, LLM re-ranking; passages feed full-text screening and extraction (`agents/screening.py`, `agents/extraction.py`) |
| 5 | Question or topic in, full PRISMA review out | ✅ | `sr-review "question"` or `--topic`; app "Start a new review" |
| 6 | Sources: PubMed, Semantic Scholar, arXiv, OpenAlex, etc. | ✅ | OpenAlex, PubMed, arXiv, Semantic Scholar (`sources/`, bulk search, tested on a recorded live answer) |
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
| 24 | Retrieval agent executes searches **and manages RAG index construction/updates** | ✅ | Searches and dedup in `agents/retrieval.py`; the RAG index is built at full-text screening and reused by extraction (`agents/screening.py`, `agents/extraction.py`), because it needs the full texts fetched there; stated in the README ("How it works") |
| 25 | Screening agents (T/A + full text) with reasoning | ✅ | `agents/screening.py` |
| 26 | Extraction & appraisal agent | ✅ | `agents/extraction.py` |
| 27 | Synthesis & writing agent: narrative, tables, final report | ✅ | `agents/synthesis.py`, `reporting/` (writer: `reporting/writer.py`) |
| 28 | Critic / QA agent: consistency, bias, completeness, protocol adherence | ✅ | `agents/critic.py`: count consistency, every outcome covered, design vs protocol, unsupported claims (numbers, risk of bias), loop back to synthesis |
| 29 | LangChain / LangGraph: agent definition, tool calling, memory, routing, state | ✅ | Agents (graph nodes), tool calling (critic, `bind_tools`), memory (SQLite checkpoint), routing (coordinator + conditional edges + guardrail), state (`state.py`) |
| 30 | RAG: hybrid (dense + sparse), re-ranking, context compression, vector store | ✅ | FAISS (dense) + BM25 (sparse) fused by reciprocal rank fusion (`rag/index.py`), LLM re-ranking (`rag/rerank.py`), context compression of each passage to the sentences that match the question (`rag/compress.py`, setting `rag_compress`, off by default because screening and extraction were evaluated without it) |
| 31 | LLM backbone: high-capability, long context | ✅ | Model configurable per agent (`config.yaml` `models:`, `SR_MODEL_<AGENT>`); runs used gpt-4o-mini (128k context) for cost, stronger models tested for the adjudicator; stated in the README |
| 32 | Tools: search APIs, PDF parsing, citation managers, PRISMA checklist validators, export (Markdown, DOCX, BibTeX) | ✅ | APIs (`sources/`), PDF and PMC XML (`sources/fulltext.py`), BibTeX, Markdown, DOCX (`reporting/`); PRISMA flow-diagram arithmetic and a PRISMA 2020 checklist validator (`reporting/prisma_checklist.py`: 42 items, appendix of the article; T2 article: 30 reported, 4 partly, 8 not reported: 6 analyses the pipeline does not do, funding and competing interests left to the authors) |
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
| 43 | README documenting all the steps followed during development and setup (install, configuration, execution, usage) | ✅ | README: "How it was built (development steps)", Install, Configure, Run, Evaluate, Use the app; full log in `docs/PLAN.md` |
| 44 | Video, 3 minutes max | ❌ | Script ready (kept locally); **to record** (user) |
| 45 | Individual contribution in the final report | ⚠️ | Report §7 drafted; **the author checks it and writes it in their own words** (user) |

## Gaps and fixes

| Gap | What was missing | Done |
|---|---|---|
| A | LLM tool calling | ✅ Critic as a LangChain tool-calling agent |
| B | Development steps in the README | ✅ "How it was built" section |
| C | PRISMA 2020 checklist validator | ✅ `reporting/prisma_checklist.py`, article appendix D |
| D | Context compression | ✅ `rag/compress.py`, setting `rag_compress` (off by default) |
| E | A real run on a free question (not an evaluation topic) | ⏳ user: one run from the app, about $0.50 |
| F | Semantic Scholar | ✅ connector |
| G | Where the RAG index is built | ✅ README |
| H | Model choice | ✅ README |
| I | Blind accuracy | ⏳ user (optional): 30 blind labels in the app |
| — | Video; individual contribution text | ⏳ user |
