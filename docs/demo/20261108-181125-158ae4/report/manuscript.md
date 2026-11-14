# Lightweight Machine-Learning Models for ECG Classification

*Systematic review report, run 20261108-181125-158ae4. Outcomes: abnormal versus normal ECG, accuracy, efficiency.*

## Abstract

**Question.** Which lightweight machine-learning models detect abnormal versus normal ECG on resource-constrained devices (microcontroller, wearable, smartphone, FPGA), and how accurate and efficient are they?

**Methods.** PRISMA 2020 review run by LLM agents with code checks; databases openalex, pubmed; designs not restricted.

**Results.** 618 records identified, 98 screened, 55 studies included. Synthesis: O1 narrative (SWiM); O2 narrative (SWiM); O3 narrative (SWiM).

**Quality check.** Critic passed (0 blockers, 1 warnings).

## Methods

### Protocol and eligibility criteria

Review question: Which lightweight machine-learning models detect abnormal versus normal ECG on resource-constrained devices (microcontroller, wearable, smartphone, FPGA), and how accurate and efficient are they?

| PICO |  |
|---|---|
| Population | ECG signals or heartbeats |
| Intervention / exposure | lightweight machine-learning models |
| Comparator | no intervention, usual care or any control |
| Outcomes | abnormal versus normal ECG; accuracy; efficiency |

Study designs: no restriction. Years: 2021 to any. Languages: English. Criteria source: topics.yaml T2 (review none (new review; no published systematic review asks this, searched 2026-11-08)).

- **D** A primary study reporting its own results, any design (reviews, meta-analyses incl. network meta-analyses, study protocols without results, editorials and letters do not meet D)
- **I1** Machine-learning model that classifies ECG signals or heartbeats
- **I2** The model separates abnormal from normal ECG (binary output, or abnormal classes reported against normal)
- **I3** Model evaluated for or on a resource-constrained device (microcontroller, wearable, smartphone or FPGA)
- **I4** At least one efficiency metric reported (model size, memory, latency, power or energy) *(full text only)*
- **I5** Peer-reviewed journal or conference publication

The protocol was drafted by the Planner agent and fixed before the search (human approval step unless the run used `--auto`).

### Information sources and search strategy

Databases: openalex, pubmed (Relevant for health and machine learning research.). The LLM proposes the search concepts; code builds each database's Boolean query and design filter, checks the hit counts before any download, and refines at most twice.

| Database | Searched (UTC) | Hits | Downloaded | Not downloaded | Query |
|---|---|---|---|---|---|
| openalex | 2026-11-08T18:11:25+00:00 | 521 | 521 | 0 | (ECG OR electrocardiogram OR heartbeats OR "cardiac signals" OR "heart rhythm") AND ("lightweight models" OR "machine learning" OR ML OR AI OR "neural networks" OR "classification models") AND (microcontroller OR MCU OR embedded OR "edge device" OR "edge computing" OR "edge AI" OR wearable OR smartphone OR "mobile phone" OR FPGA OR ASIC OR SoC OR "system on chip" OR TinyML OR low-power OR "low power" OR quantized OR quantization OR on-device OR resource-constrained) AND ("abnormal ECG" OR "normal ECG" OR "ECG classification" OR "arrhythmia detection" OR "ECG diagnosis") |
| pubmed | 2026-11-08T18:11:25+00:00 | 97 | 97 | 0 | (ECG[tiab] OR electrocardiogram[tiab] OR heartbeats[tiab] OR "cardiac signals"[tiab] OR "heart rhythm"[tiab] OR Electrocardiography[mh]) AND ("lightweight models"[tiab] OR "machine learning"[tiab] OR ML[tiab] OR AI[tiab] OR "neural networks"[tiab] OR "classification models"[tiab] OR "Machine Learning"[mh]) AND (microcontroller[tiab] OR MCU[tiab] OR embedded[tiab] OR "edge device"[tiab] OR "edge computing"[tiab] OR "edge AI"[tiab] OR wearable[tiab] OR smartphone[tiab] OR "mobile phone"[tiab] OR FPGA[tiab] OR ASIC[tiab] OR SoC[tiab] OR "system on chip"[tiab] OR TinyML[tiab] OR low-power[tiab] OR "low power"[tiab] OR quantized[tiab] OR quantization[tiab] OR on-device[tiab] OR resource-constrained[tiab]) AND ("abnormal ECG"[tiab] OR "normal ECG"[tiab] OR "ECG classification"[tiab] OR "arrhythmia detection"[tiab] OR "ECG diagnosis"[tiab]) |

### Selection process

Records were deduplicated by code (record id, DOI, PMID, PMCID, then fuzzy title match with guards). Only the 100 best-ranked unique records were screened (screening limit; rank = best position in any database's relevance order, so every database contributes its top hits); the other 455 are reported in the flow diagram as not screened. Automation tool (PRISMA 2020 'records marked as ineligible by automation tools'): before any LLM call, a code rule excludes on the design criterion D a record whose title names its own type as a review, meta-analysis, protocol, editorial or commentary (at the start of the title or after a colon or dash, e.g. '...: a systematic review'), and, when a criterion requires peer review, a record with an arXiv DOI (10.48550) is excluded on it; these records are counted before screening, not as screening exclusions. Two LLM screeners (A strict, B lenient) judged every record criterion by criterion, each failure backed by a quote that code verifies in the text; disagreements went to an LLM adjudicator, unresolved records to a human (or were included with `--auto`). Full texts (open access only: PMC XML, open PDFs) were indexed (FAISS + BM25, LLM re-ranking) and the screeners read the best passages per criterion; papers without a full text were assessed on the abstract and flagged.

Screening rules were tuned on a development sample of the T3 topic and then frozen (no change after seeing the held-out topics, except reported bug fixes). Changes made during tuning:

| Part | Change |
|---|---|
| 6 | Judge what the study did; background sentences are not evidence; year and language criteria never decide |
| 6 | Screener B excludes only on a clearly broken criterion; the adjudicator never excludes or leaves unresolved for missing information at title/abstract |
| 6 | At full text without a full text: no exclusion for what the abstract does not report |
| 6 | Eligibility never depends on results |
| 6 | Design criterion D = primary study (not reviews, meta-analyses, protocols) |
| 6b | Evaluation runs use the gold review's criteria (topics.yaml), not the Planner's draft |
| 6b | One verdict per criterion; deciding criterion chosen by code (D, I1.., E1..); a failure needs a verified quote |
| 6b | Design guard: the adjudicator cannot include a verified D failure without a verified D 'yes' |
| 6b | D as one positive sentence, same yes/no/unclear definition for both screeners, quote before the verdict, 'pools other studies -> D no' |
| 6b | Output cap of 4,096 tokens per call (one call looped to 16,384) |
| 6b | Contradicting-quote check: a D 'no' quoting a listed design without a non-primary word becomes unclear |
| 6c | Criteria tagged full text never decide at title/abstract |
| 6c | Multi-arm trials judged on the eligible arm pair (comparison field) |
| 6c | Exclusions on different criteria go to the adjudicator; positive test sentences for exclusion criteria |
| 6d | Title rule for D (protocol, systematic review, meta-analysis... where the title names the record type) |
| 6d | Eligibility-list quotes: an exclusion 'yes' quoted from the study's own eligibility sentence becomes unclear |
| 6d | Adjudicator model kept at gpt-4o-mini (gpt-4.1-mini tested: changes were worse) |

### Data collection and risk of bias

Fields were derived from the protocol (PICO, design, sample size, outcomes and results). A value was kept only with a quote found in the passages shown and every number of the value printed in that quote; otherwise it is `not reported` (values are never computed or invented). Risk of bias tool chosen by code from the verified design: RoB 2 (randomised trials, first protocol outcome), ROBINS-I (non-randomised studies), QUADAS-2 (diagnostic accuracy), else a 5-item checklist; a domain judgment needs a verified quote. This protocol replaces them with its own instrument for every study (Five-domain instrument of Arredondo-Velazquez 2026 (Table 2): D1 Evaluation protocol stated; D2 Subjects separated between training and test; D3 Per-class or prevalence-insensitive metric; D4 Deployed variant evaluated; D5 Model size reported).

### Synthesis methods

Continuous outcomes were pooled (random effects, DerSimonian-Laird; Hedges' g from post-intervention means) only with at least 3 studies reporting verified per-arm n, mean and SD; the 95% prediction interval is given when k >= 3, and a pooled result with I2 >= 75% is reported as not interpretable as a single effect. Otherwise the synthesis follows SWiM (Campbell 2020): grouping by outcome, direction of effect as the standardised metric, vote counting with a sign test, heterogeneity described. Only between-group results give a direction; the metric order was pre-specified (between group estimate, post means, change means), and a study whose metrics disagree is flagged. Diagnostic studies: sensitivity and specificity with Wilson 95% CIs; pooled (univariate logit random effects) with at least 3 studies. A critic agent checked the text (code checks plus an LLM pass); blockers sent it back to synthesis at most 2 times. An LLM objection counted as a blocker only when code confirmed the wrong value.

### Automation and reproducibility

Agents and models: adjudicator: gpt-4o-mini; coordinator: gpt-4o-mini; critic: gpt-4o-mini; extraction: gpt-4o-mini; planner: gpt-4o-mini; reranker: gpt-4o-mini; risk_of_bias: gpt-4o-mini; screener_a: gpt-4o-mini; screener_b: gpt-4o-mini; search_strategy: gpt-4o-mini; synthesis: gpt-4o-mini. Temperature 0. Every LLM answer is cached on disk, keyed on (model, output schema, prompt): re-running the review with the same cache reproduces every decision at no cost. In this run 228 of 1375 LLM calls came from the cache. A fresh run without the cache can differ: the model is not deterministic across calls (seen during development: a re-extraction lost one study's table result and gained another's). Database answers are cached the same way; the search dates are those listed above.

## Results

### Study selection

618 records were identified; 63 duplicates, 2 records marked as ineligible by automation tools (title rule, arXiv rule), 0 records not downloaded (search ceiling) and 455 records above the screening limit of 100 (not screened) were removed before screening. 98 records were screened on title and abstract and 28 excluded. 70 reports were sought; 40 had no open-access full text and were assessed on the abstract. 15 were excluded and **55 studies were included**. Screener agreement: title/abstract kappa 0.0568, full text kappa 0.7429.

![PRISMA 2020 flow diagram](prisma_flow.png)

### Study characteristics

| Study | Design | Country | n | Population | Intervention | Comparator | Text used |
|---|---|---|---|---|---|---|---|
| S1 Sundar 2026 | not reported | not reported | not reported | not reported | Edge AI architecture specifically for real-time arrhythmia identification using wearable ECG devices. | centralized cloud computing architecture | abstract only † |
| S2 Farag 2023 | Tiny Matched Filter-Based CNN for Inter-Patient ECG Classification and Arrhythmia Detection at the Edge. | not reported | 47 | not reported | Tiny convolutional neural network (CNN) classifier. | Usual care. | full text |
| S3 Fan 2024 | not reported | not reported | not reported | ECG arrhythmia detection | time domain (TD) based SNN processor for ECG classification | no intervention | abstract only † |
| S4 Zhang 2023 | not reported | not reported | not reported | not reported | not reported | not reported | abstract only † |
| S5 SHANMUGA SUNDARI M 2026 | not reported | not reported | not reported | ECG signals or heartbeats | lightweight 1D CNN | usual care | abstract only † |
| S6 Rahman 2025 | not reported | not reported | not reported | not reported | not reported | not reported | full text |
| S7 Wang 2021 | not reported | not reported | not reported | ECG signals or heartbeats | lightweight machine-learning models for ECG signal diagnosis | not reported | abstract only † |
| S8 Pillai 2025 | not reported | not reported | not reported | ECG signals for arrhythmia detection | arrDWNN model for arrhythmia detection from ECG signals | not reported | abstract only † |
| S9 Zahid 2023 | not reported | not reported | not reported | ECG signals or heartbeats | 1D Self-ONN with feature injection for ECG classification | not reported | abstract only † |
| S10 Kang 2025 | not reported | not reported | not reported | ECG signals or heartbeats | Range Encoding: a low-power spiking neural network encoding scheme | not reported | abstract only † |
| S11 Okoror 2026 | lightweight neural-network approach for binary ECG beat classification | not reported | 84,708 | not reported | lightweight neural-network approach for binary ECG beat classification | no intervention | full text |
| S12 Nandini 2025 | not reported | not reported | not reported | ECG signals | FPGA hardware acceleration with machine learning | traditional software implementations | full text |
| S13 Lal 2023 | not reported | not reported | not reported | Individuals with congestive heart failure (CHF), cardiac arrhythmia (ARR), and normal sinus rhythms (NSR) | Compressed Learning (CL) to detect abnormal ECGs | not reported | abstract only † |
| S14 Sancho 2023 | not reported | not reported | not reported | ECG signals or heartbeats; health status: arrhythmias. | deep learning arrhythmia detection model | not reported | abstract only † |
| S15 Tran 2026 | not reported | not reported | not reported | not reported | 1D CNN for ECG classification | not reported | abstract only † |
| S16 Vinod 2026 | intelligent edge device for real time cardiac arrhythmia detection | not reported | not reported | Cardiac arrhythmias | not reported | usual care | full text |
| S17 J 2026 | not reported | not reported | not reported | ECG signals or heartbeats | lightweight deep learning model | not reported | abstract only † |
| S18 Bhandari 2025 | not reported | not reported | not reported | ECG signals or heartbeats | lightweight SNN architecture based on Leaky Integrate and Fire neurons | no intervention | abstract only † |
| S19 Thanh 2026 | not reported | not reported | not reported | ECG signals for early detection of cardiac arrhythmias. | lightweight Edge AI framework for real-time arrhythmia classification using CWT scalogram representations and optimized EfficientNet-B0 architecture. | no intervention. | abstract only † |
| S20 Shrirao 2026 | implementation, design, and demonstration of an ultra-low-power embedded processor | India | not reported | healthy subjects | lightweight machine learning inference including arrhythmia detection and temperature anomaly classification | no intervention | full text |
| S21 Han 2023 | not reported | not reported | not reported | not reported | not reported | not reported | abstract only † |
| S22 Mukhopadhyay 2023 | not reported | not reported | not reported | not reported | not reported | not reported | abstract only † |
| S23 RYO 2026 | experimental research | not reported | 48 | not reported | lightweight one-dimensional Convolutional Neural Network (1D-CNN) optimized for resource-constrained embedded platforms | not reported | full text |
| S24 Banjo 2026 | Cascaded Quantized Spiking Neural Network for Real-Time ECG Arrhythmia Detection on Edge Hardware. | not reported | 47 | ECG signals or heartbeats. | quantized convolutional spiking neural network (QCSNN) for real-time arrhythmia detection. | No intervention. | full text |
| S25 Geng 2025 | not reported | not reported | not reported | not reported | not reported | not reported | abstract only † |
| S26 Devi 2025 | not reported | not reported | not reported | ECG signals or heartbeats | Edge-Adaptive Temporal-Spatial Neural Engine (EATNE) | not reported | abstract only † |
| S27 Gonawala 2025 | not reported | not reported | not reported | not reported | not reported | not reported | abstract only † |
| S28 Pham 2025 | not reported | not reported | not reported | not reported | not reported | usual care or any control. | abstract only † |
| S29 Meng 2026 | not reported | not reported | not reported | ECG signals or heartbeats | low-parameter, high-precision spiking neural network (SNN) for real-time ECG classification | not reported | abstract only † |
| S30 Cong 2024 | not reported | not reported | not reported | not reported | TinyML-based system for ECG rhythm classification using a microcontroller. | not reported | abstract only † |
| S31 Hammad 2024 | not reported | not reported | not reported | ECG signals or heartbeats | K-Nearest Neighbors (KNN), Random Forest (RF), and Logistic Regression (LR) for ECG classification | not reported | abstract only † |
| S32 Tao 2023 | not reported | not reported | not reported | ECG signals or heartbeats | lightweight machine-learning models | no intervention | abstract only † |
| S33 Chen 2024 | not reported | not reported | not reported | ECG signals | EMCNet, Ensemble Multiscale Convolutional Neural Network for single-lead ECG classification | not reported | abstract only † |
| S34 Rosengarten 2023 | not reported | not reported | 3505 | Patients referred for investigation of heart rhythm symptoms. | AI-led remote ECG monitoring service (Zio XT) | standard Holter monitoring | abstract only † |
| S35 Lee 2025 | not reported | not reported | 5000 | ECG signals or heartbeats | 1-D convolutional neural network | not reported | abstract only † |
| S36 Ismail 2023 | 1D Temporal Convolutional Network (TCN) based architecture | not reported | not reported | ECG signals for heart disease diagnosis | Temporal Convolutional Network (TCN) for ECG classification | not reported | full text |
| S37 Sadeghi 2024 | not reported | not reported | not reported | ECG signals or heartbeats; seven distinct heart states. | 3DECG-Net, a deep learning model for ECG analysis. | not reported | abstract only † |
| S38 Panwar 2025 | Integrated portable ECG monitoring system with CNN classification for early arrhythmia detection. | not reported | 47 | ECG signals. | Convolutional Neural Networks (CNNs) for accurate classification of cardiac abnormalities. | No intervention. | full text |
| S39 Deng 2024 | Instruction Driven 1-D CNN Processor | not reported | not reported | ECG signals from patients with various heart conditions. | Instruction-driven convolutional neural network (CNN) processor. | not reported | full text |
| S40 Talukder 2025 | explainable deep learning framework | not reported | 47 | ECG signals from subjects with various arrhythmias. | explainable deep learning framework (CNN and DNN) for arrhythmia detection. | no intervention. | full text |
| S41 Singh 2026 | not reported | not reported | not reported | not reported | linear support vector classifier | not reported | abstract only † |
| S42 Mantravadi 2024 | not reported | not reported | not reported | ECG signals or heartbeats | CLINet deep learning network for ECG signal classification | not reported | abstract only † |
| S43 Khalid 2026 | Lightweight and interpretable edge intelligence AI with intrusion detection for trustworthy cardiac arrhythmia in medical IoT. | not reported | 12 | ECG signals or heartbeats | CLARITY-AI 2.0, a lightweight and trustworthy arrhythmia detection framework for MIoT. | usual care | full text |
| S44 Li 2023 | not reported | not reported | not reported | ECG recordings from wearable devices; atrial fibrillation detection. | dilated residual neural network; trained on decomposed ECG signals. | not reported | abstract only † |
| S45 Wu 2025 | not reported | not reported | 47 | not reported | CTM-based all-hardware RC system for energy-efficient arrhythmia classification. | not reported | full text |
| S46 Dózsa 2024 | not reported | not reported | not reported | ECG data | variable projection support vector machine (VP-SVM) algorithm | not reported | abstract only † |
| S47 Faraone 2021 | not reported | not reported | not reported | not reported | Knowledge Distillation applied to convolutional-recurrent neural network for cardiac arrhythmia detection and classification. | not reported | abstract only † |
| S48 Yang 2025 | not reported | not reported | not reported | ECG signals | geometric framework for robust ECG classification using PSR and DTM | not reported | abstract only † |
| S49 Cao 2025 | not reported | not reported | not reported | ECG diagnosis | lightweight MambaOut architecture for ECG diagnosis | no intervention | abstract only † |
| S50 Qiang 2024 | not reported | not reported | not reported | not reported | Convolutional Residual Graph Neural Network (Conv-RGNN) for ECG classification. | not reported | abstract only † |
| S51 Thapa 2025 | progressive deep learning framework | not reported | not reported | not reported | not reported | no intervention | full text |
| S52 Sahu 2022 | not reported | not reported | not reported | not reported | LTH-ECG, a novel goal-driven winning lottery ticket discovery method for ECG classification. | not reported | abstract only † |
| S53 Tang 2026 | Lightweight element-wise product enhanced neural network for efficient arrhythmia detection on embedded devices. | not reported | not reported | ECG signals or heartbeats; early and accurate arrhythmia detection. | lightweight model optimized for real-time arrhythmia detection. | no intervention. | abstract only † |
| S54 Menon 2023 | automated morphological and clinical analyses of ECGs are well documented | United States | 10 | ICU patients | template matching approach for signal quality assessment on ECG data | no intervention | full text |
| S55 Yaser 2025 | not reported | not reported | not reported | ECG signals; patients with cardiovascular diseases. | Temporal Convolutional Networks (TCN) and Graph Convolutional Networks (GCN) for cardiac arrhythmia detection. | not reported | abstract only † |

### Risk of bias

Overall: unclear 46, high 9.

| Study | Tool | Overall | Domains |
|---|---|---|---|
| S1 Sundar 2026 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S2 Farag 2023 | checklist | unclear | D1 low; D2 low; D3 low; D4 unclear; D5 low |
| S3 Fan 2024 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S4 Zhang 2023 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S5 SHANMUGA SUNDARI M 2026 | checklist | unclear | D1 unclear; D2 unclear; D3 low; D4 unclear; D5 unclear |
| S6 Rahman 2025 | checklist | unclear | D1 low; D2 low; D3 unclear; D4 unclear; D5 unclear |
| S7 Wang 2021 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S8 Pillai 2025 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S9 Zahid 2023 | checklist | unclear | D1 unclear; D2 unclear; D3 low; D4 unclear; D5 unclear |
| S10 Kang 2025 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S11 Okoror 2026 | checklist | unclear | D1 low; D2 low; D3 low; D4 low; D5 unclear |
| S12 Nandini 2025 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S13 Lal 2023 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S14 Sancho 2023 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S15 Tran 2026 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S16 Vinod 2026 | checklist | high | D1 unclear; D2 unclear; D3 high; D4 low; D5 unclear |
| S17 J 2026 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S18 Bhandari 2025 | checklist | unclear | D1 low; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S19 Thanh 2026 | checklist | high | D1 unclear; D2 unclear; D3 unclear; D4 high; D5 unclear |
| S20 Shrirao 2026 | checklist | unclear | D1 low; D2 unclear; D3 unclear; D4 unclear; D5 low |
| S21 Han 2023 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S22 Mukhopadhyay 2023 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S23 RYO 2026 | checklist | high | D1 unclear; D2 low; D3 high; D4 unclear; D5 low |
| S24 Banjo 2026 | checklist | high | D1 low; D2 high; D3 low; D4 low; D5 low |
| S25 Geng 2025 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S26 Devi 2025 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S27 Gonawala 2025 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 low |
| S28 Pham 2025 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 low |
| S29 Meng 2026 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S30 Cong 2024 | checklist | unclear | D1 unclear; D2 unclear; D3 low; D4 unclear; D5 low |
| S31 Hammad 2024 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S32 Tao 2023 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S33 Chen 2024 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S34 Rosengarten 2023 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S35 Lee 2025 | checklist | unclear | D1 low; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S36 Ismail 2023 | checklist | unclear | D1 low; D2 low; D3 low; D4 unclear; D5 low |
| S37 Sadeghi 2024 | checklist | high | D1 unclear; D2 low; D3 high; D4 unclear; D5 unclear |
| S38 Panwar 2025 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 low |
| S39 Deng 2024 | checklist | unclear | D1 unclear; D2 unclear; D3 low; D4 unclear; D5 low |
| S40 Talukder 2025 | checklist | unclear | D1 unclear; D2 unclear; D3 low; D4 unclear; D5 low |
| S41 Singh 2026 | checklist | high | D1 unclear; D2 low; D3 high; D4 unclear; D5 low |
| S42 Mantravadi 2024 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 low |
| S43 Khalid 2026 | checklist | unclear | D1 low; D2 unclear; D3 low; D4 unclear; D5 low |
| S44 Li 2023 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S45 Wu 2025 | checklist | unclear | D1 low; D2 low; D3 low; D4 unclear; D5 low |
| S46 Dózsa 2024 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S47 Faraone 2021 | checklist | unclear | D1 unclear; D2 unclear; D3 low; D4 unclear; D5 low |
| S48 Yang 2025 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S49 Cao 2025 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 low |
| S50 Qiang 2024 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S51 Thapa 2025 | checklist | high | D1 unclear; D2 high; D3 low; D4 unclear; D5 unclear |
| S52 Sahu 2022 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |
| S53 Tang 2026 | checklist | high | D1 unclear; D2 unclear; D3 high; D4 low; D5 low |
| S54 Menon 2023 | checklist | high | D1 unclear; D2 low; D3 high; D4 unclear; D5 unclear |
| S55 Yaser 2025 | checklist | unclear | D1 unclear; D2 unclear; D3 unclear; D4 unclear; D5 unclear |

### Synthesis of results

**O1: abnormal versus normal ECG** (better = unknown values). Method: narrative (SWiM) (0 study(ies) with per-arm n, mean and SD (< 3 needed for a meta-analysis): vote counting by direction of effect). Vote count: favours intervention 0, favours control 0, no difference 0, not extractable 55, sign test p n/a.

**O2: accuracy** (better = higher values). Method: narrative (SWiM) (0 study(ies) with per-arm n, mean and SD (< 3 needed for a meta-analysis): vote counting by direction of effect). Vote count: favours intervention 2, favours control 0, no difference 0, not extractable 53, sign test p 0.500.

**O3: efficiency** (better = higher values). Method: narrative (SWiM) (0 study(ies) with per-arm n, mean and SD (< 3 needed for a meta-analysis): vote counting by direction of effect). Vote count: favours intervention 0, favours control 0, no difference 0, not extractable 55, sign test p n/a.

Narrative synthesis (written by the synthesis agent, every number and study label checked by code):

#### O1: abnormal versus normal ECG

The studies were grouped based on their reported outcomes related to abnormal versus normal ECG classification, with no studies providing sufficient data for a meta-analysis, leading to a vote counting by direction of effect [S1†, S2, S3†, S4†, S5†, S6, S7†, S8†, S9†, S10†, S11, S12, S13†, S14†, S15†, S16, S17†, S18†, S19†, S20, S21†, S22†, S23, S24, S25†, S26†, S27†, S28†, S29†, S30†, S31†, S32†, S33†, S34†, S35†, S36, S37†, S38, S39, S40, S41†, S42†, S43, S44†, S45, S46†, S47†, S48†, S49†, S50†, S51, S52†, S53†, S54, S55†].

*Heterogeneity.* The studies differ in their reported measures, with some focusing on ECG classification accuracy, while others report various performance metrics without clear direction of effect [S1†, S2, S3†, S4†, S5†, S6, S7†, S8†, S9†, S10†, S11, S12, S13†, S14†, S15†, S16, S17†, S18†, S19†, S20, S21†, S22†, S23, S24, S25†, S26†, S27†, S28†, S29†, S30†, S31†, S32†, S33†, S34†, S35†, S36, S37†, S38, S39, S40, S41†, S42†, S43, S44†, S45, S46†, S47†, S48†, S49†, S50†, S51, S52†, S53†, S54, S55†].

#### O2: accuracy

The studies were grouped based on their reported accuracy metrics, with no studies providing sufficient data for a meta-analysis, leading to a vote counting by direction of effect [S1†, S2, S3†, S4†, S5†, S6, S7†, S8†, S9†, S10†, S11, S12, S13†, S14†, S15†, S16, S17†, S18†, S19†, S20, S21†, S22†, S23, S24, S25†, S26†, S27†, S28†, S29†, S30†, S31†, S32†, S33†, S34†, S35†, S36, S37†, S38, S39, S40, S41†, S42†, S43, S44†, S45, S46†, S47†, S48†, S49†, S50†, S51, S52†, S53†, S54, S55†].

*Heterogeneity.* The studies differ in their reported accuracy metrics, with some providing accuracy percentages while others report sensitivity and specificity without a clear direction of effect [S1†, S2, S3†, S4†, S5†, S6, S7†, S8†, S9†, S10†, S11, S12, S13†, S14†, S15†, S16, S17†, S18†, S19†, S20, S21†, S22†, S23, S24, S25†, S26†, S27†, S28†, S29†, S30†, S31†, S32†, S33†, S34†, S35†, S36, S37†, S38, S39, S40, S41†, S42†, S43, S44†, S45, S46†, S47†, S48†, S49†, S50†, S51, S52†, S53†, S54, S55†].

#### O3: efficiency

The studies were grouped based on their reported efficiency metrics, with no studies providing sufficient data for a meta-analysis, leading to a vote counting by direction of effect [S1†, S2, S3†, S4†, S5†, S6, S7†, S8†, S9†, S10†, S11, S12, S13†, S14†, S15†, S16, S17†, S18†, S19†, S20, S21†, S22†, S23, S24, S25†, S26†, S27†, S28†, S29†, S30†, S31†, S32†, S33†, S34†, S35†, S36, S37†, S38, S39, S40, S41†, S42†, S43, S44†, S45, S46†, S47†, S48†, S49†, S50†, S51, S52†, S53†, S54, S55†].

*Heterogeneity.* The studies differ in their reported efficiency metrics, with some focusing on energy consumption while others report latency and computational efficiency without a clear direction of effect [S1†, S2, S3†, S4†, S5†, S6, S7†, S8†, S9†, S10†, S11, S12, S13†, S14†, S15†, S16, S17†, S18†, S19†, S20, S21†, S22†, S23, S24, S25†, S26†, S27†, S28†, S29†, S30†, S31†, S32†, S33†, S34†, S35†, S36, S37†, S38, S39, S40, S41†, S42†, S43, S44†, S45, S46†, S47†, S48†, S49†, S50†, S51, S52†, S53†, S54, S55†].

#### Overall

The evidence is limited by the high number of studies with unclear risk of bias (46 out of 55), and the majority of studies were extracted from abstracts, which may affect the reliability of the findings [S1†, S2, S3†, S4†, S5†, S6, S7†, S8†, S9†, S10†, S11, S12, S13†, S14†, S15†, S16, S17†, S18†, S19†, S20, S21†, S22†, S25†, S26†, S27†, S28†, S29†, S30†, S31†, S32†, S33†, S34†, S35†, S36, S37†, S38, S39, S40, S41†, S42†, S43, S44†, S45, S46†, S47†, S48†, S49†, S50†, S51, S52†, S53†, S54, S55†].

† Data extracted from the abstract only (no open-access full text).

## Quality assurance (critic)

Critic round 1 on synthesis round 1: passed, 0 blockers, 1 warnings; loops used 0/2.

Warnings:

- `llm_unsupported_claim`: sentence 4 ('The evidence is limited by the high number of studies with unclear risk of bias (46 out of 55), and the majority of stud'): The sentence states that 46 out of 55 studies have unclear risk of bias, but the evidence indicates that there are 46 studies with unclear risk of bias and 9 studies with high risk of bias, which is not mentioned in the sentence. (LLM objection; evidence quote not found in the tables; code check: no checkable value named)


## Limitations

- Databases searched: openalex, pubmed only (free sources); reviews that also search Embase, Cochrane or Scopus may find studies these do not index.
- Screening limit: only the 100 best-ranked of 555 unique records were screened; relevant studies among the other 455 were not considered.
- 39 of 55 included studies had no open-access full text and were screened and extracted from the abstract only (S1, S3, S4, S5, S7, S8, S9, S10, S13, S14, S15, S17, S18, S19, S21, S22, S25, S26, S27, S28, S29, S30, S31, S32, S33, S34, S35, S37, S41, S42, S44, S46, S47, S48, S49, S50, S52, S53, S55, marked †).
- Extraction values need a verified quote: correct values paraphrased by the model are refused and reported as `not reported` (conservative, lowers completeness).
- Risk of bias (RoB 2) was assessed for the first protocol outcome only; diagnostic pooling is univariate (not the bivariate model).
- Screening, extraction and synthesis are done by LLM agents with code checks; a human checked only the steps where the run paused for approval.

## Run summary

- Run `20261108-181125-158ae4`, 13.5 min wall-clock (includes time waiting for a human at the approval steps).
- Coordinator steps: 9; decisions accepted 8, refused 1.
- LLM calls: 1375 (228 from the cache), 2,957,454 input and 287,249 output tokens.
- API cost: $0.5618, of which embeddings $0.0046.

| Agent / node | Calls | Cached | Input tokens | Output tokens | Cost (USD) |
|---|---|---|---|---|---|
| extraction | 770 | 159 | 1,285,587 | 152,172 | 0.2535 |
| ft_screening | 328 | 21 | 1,213,653 | 65,605 | 0.2118 |
| ta_screening | 264 | 44 | 424,195 | 66,566 | 0.0863 |
| synthesis | 1 | 0 | 12,912 | 1,467 | 0.0028 |
| critic | 1 | 0 | 13,277 | 262 | 0.0021 |
| coordinator | 9 | 3 | 4,618 | 339 | 0.0007 |
| planner | 1 | 1 | 2,037 | 528 | 0.0000 |
| search_strategy | 1 | 0 | 1,175 | 310 | 0.0000 |

Coordinator proposals refused by the guardrail (deterministic substitute run instead):

- step 5: proposed `ta_screening`, ran `ft_screening` (ta_screening is already up to date: its inputs have not changed)

## Included studies

- **S1** R. Sundar, Avinash Deyyam, Anusha N et al. (2026). Edge AI Architecture for Real-Time Arrhythmia Detection Using Wearable ECG Devices. International Conference Trends Material Science and Inventive Materials doi:10.1109/ictmim68190.2026.11507301
- **S2** Mohammed M Farag (2023). A Tiny Matched Filter-Based CNN for Inter-Patient ECG Classification and Arrhythmia Detection at the Edge. Sensors (Basel, Switzerland) doi:10.3390/s23031365
- **S3** Haodong Fan, Liang Chang, Junlu Zhou et al. (2024). An Ultra-Low Power Time-Domain based SNN Processor for ECG Classification. IEEE International Symposium on Circuits and Systems proceedings doi:10.1109/iscas58744.2024.10558014
- **S4** Shuo Zhang, Qianfang Sun, Lingfeng Yao et al. (2023). Feature Detection Algorithm Combined with Machine Learning Applied to Abnormal ECG Diagnosis System. Smart innovation, systems and technologies doi:10.1007/978-981-19-7184-6_17
- **S5** SHANMUGA SUNDARI M , T. LAKSHMI PRAVEENA , PUSHPA BHUPATIRAJU , ASHA VUYYURU , BUKKACHARLA KISHORE KUMAR , B. GOVARDHAN SIVA SAI (2026). REAL-TIME ARRHYTHMIA DETECTION FROM WEARABLE ECG DEVICES USING LIGHTWEIGHT 1D CNN AND EDGE AI DEPLOYMENT. Journal of Theoretical and Applied Information Technology doi:10.5281/zenodo.18666630
- **S6** Mahfuzur Rahman, Bashir I. Morshed (2025). Resource-Constrained On-Chip AI Classifier for Beat-by-Beat Real-Time Arrhythmia Detection with an ECG Wearable System. Electronics doi:10.3390/electronics14132654
- **S7** Bocheng Wang, Guorong Chen, Lu Rong et al. (2021). ECG diagnosis device based on machine learning. IEEE International Conference on Emergency Science and Information Technology (ICESIT) doi:10.1109/icesit53460.2021.9697057
- **S8** Velu Pillai, Shashank Nag, Mugdha Jadhao et al. (2025). Edge-Optimized Weightless Neural Network for Low-Power Wearable Arrhythmia Detection. IEEE Dallas Circuits and Systems Conference (DCAS) doi:10.1109/dcas65331.2025.11045471
- **S9** Muhammad Uzair Zahid, Serkan Kiranyaz, Moncef Gabbouj (2023). Global ECG Classification by Self-Operational Neural Networks With Feature Injection. IEEE transactions on bio-medical engineering doi:10.1109/tbme.2022.3187874
- **S10** Xinyu Kang, Xingbo Wang, Yuru Li et al. (2025). Range Encoding: A Low-Power Spiking Neural Network Encoding Scheme for ECG Classification. IEEE International Conference on Artificial Intelligence Circuits and Systems (AICAS) doi:10.1109/aicas64808.2025.11173111
- **S11** Mclawrence Okoror (2026). TinyML-Based ECG Abnormality Detection for Resource-Constrained Embedded Systems.  doi:10.31224/8569
- **S12** K. P. Nandini, G. Seshikala (2025). Efficient ECG Arrhythmia Detection on FPGA using Machine Learning and Fiducial Windowing. Engineering Technology & Applied Science Research doi:10.48084/etasr.9589
- **S13** Bharat Lal, Qimeng Li, Pasquale Corsonello et al. (2023). Abnormal ECG Detection in Wearable Devices Using Compressed Learning. IEEE International Conference on Networking, Sensing and Control (ICNSC) doi:10.1109/icnsc58704.2023.10319001
- **S14** Juan Manuel Sancho, Jônata Tyska Carvalho (2023). On the Impact of ECG Data Quality for Arrhythmia Detection Using Convolutional Neural Networks and Wearable Devices. IEEE Symposium Series on Computational Intelligence (SSCI) doi:10.1109/ssci52147.2023.10372028
- **S15** Tuan-Kiet Tran, Tan-Phat Dang, Nhu-Hoang Nguyen et al. (2026). Model-to-Hardware Co-Design of a Resource-Efficient RISC-V Edge AI SoC for 1D CNN ECG Classification. IEEE Embedded Systems Letters doi:10.1109/les.2026.3708151
- **S16** Joel Vinod, Kiran Bobby, Linu Jose et al. (2026). Intelligent Edge Device for Real Time Cardiac Arrhythmia Detection. INTERANTIONAL JOURNAL OF SCIENTIFIC RESEARCH IN ENGINEERING AND MANAGEMENT doi:10.55041/ijsrem57447
- **S17** Madheswari J, Gowtham K, Ananthan S et al. (2026). Smart Portable ECG Monitoring System with Edge AI for Real-Time Arrhythmia Detection. Zenodo (CERN European Organization for Nuclear Research) doi:10.5281/zenodo.19451701
- **S18** Shruti Bhandari, Batuhan Aşıroğlu, Sanjog Dhakal et al. (2025). Spiking Neural Networks for ECG Classification and Anomaly Detection. IEEE Annual Ubiquitous Computing, Electronics & Mobile Communication Conference (UEMCON) doi:10.1109/uemcon67449.2025.11267772
- **S19** Ho Xuan Thanh, Nguyen The Bach, Luong Minh An et al. (2026). Edge AI-Based ECG Arrhythmia Detection Using CWT Scalograms and Optimized EfficientNet for Real-Time Inference. International Conference on Communications and Electronics (ICCE) doi:10.1109/icce70609.2026.11658447
- **S20** Nisha Milind Shrirao, Sumit Ramswami Punam (2026). Ultra-Low-Power Embedded Processor for Wearable Healthcare Monitoring. SCCTS Journal of Embedded Systems Design and Applications doi:10.31838/esa/03.01.04
- **S21** Yibo Han, Pu Han, Bo Yuan et al. (2023). Novel Transformation Deep Learning Model for Electrocardiogram Classification and Arrhythmia Detection using Edge Computing. Journal of Grid Computing doi:10.1007/s10723-023-09717-3
- **S22** Shalini Mukhopadhyay, S. Dey, Avik Ghose et al. (2023). Generating Tiny Deep Neural Networks for ECG Classification on Micro-Controllers. IEEE International Conference on Pervasive Computing and Communications Workshops (PerCom Workshops) doi:10.1109/percomworkshops56833.2023.10150311
- **S23** RYO MIRZAVALI ADITYA RYO, Muhammad Raffa Rabbani (2026). Edge AI-Based Smart Healthcare Architecture for Real-Time Arrhythmia Detection Using Lightweight Deep Learning on Wearable Devices Without Cloud IoT. JURNAL SISTEM DAN TEKNOLOGI INFORMASI
- **S24** Olamilekan Banjo, Behnaz Ghoraani (2026). A Cascaded Quantized Spiking Neural Network for Real-Time ECG Arrhythmia Detection on Edge Hardware. Sensors (Basel, Switzerland) doi:10.3390/s26123723
- **S25** Shiyong Geng, Changhao Sun, Yongping Dan (2025). Optimized negative spikes in Spiking Neural Networks for low-latency ECG classification. Pattern Recognition Letters doi:10.1016/j.patrec.2025.04.023
- **S26** P. Syamala Devi, K. Hima Bindu, B. Hussen Basha et al. (2025). VLSI Implementation of Edge-Adaptive Temporal-Spatial Neural Engine (EATNE) for Ultra-Low-Power ECG Arrhythmia Detection. International Conference on Applied Artificial Intelligence and Computing (ICAAIC) doi:10.1109/icaaic64647.2025.11331031
- **S27** Rahul Gonawala, Priyank H. Prajapati, Anand D. Darji et al. (2025). ECG Arrhythmia Detection using SVM Classifier on FPGA. International Conference on Intelligent Systems and Embedded Design (ISED) doi:10.1109/ised67359.2025.11405230
- **S28** Hoai Luan Pham, Thi Diem Tran, Vu Trung Duong Le et al. (2025). MINA: A Hardware-Efficient and Flexible Mini-InceptionNet Accelerator for ECG Classification in Wearable Devices. IEEE Transactions on Circuits and Systems I Regular Papers doi:10.1109/tcsi.2025.3553837
- **S29** Liwei Meng, Ning Ning, Ruichen Ma et al. (2026). Efficient Spiking Neural Networks for Accurate and Real-Time ECG Classification. International Conference on Advanced Algorithms and Control Engineering (ICAACE) doi:10.1109/icaace69793.2026.11508717
- **S30** Duan Luong Cong, Bien Nguyen Quang, Loi Khanh Nguyen et al. (2024). TinyML-Based System for Cost-Effective ECG Rhythm Classification on Embedded Device. International Conference on Advanced Technologies for Communications doi:10.1109/atc63255.2024.10908303
- **S31** Anfal Hamid Hammad, Azmi Shawkat Abdulbaqi (2024). 2D ECG classification system based on machine learning and LBP. AIP conference proceedings doi:10.1063/5.0190893
- **S32** Rui Tao, Lin Wang, Binrong Wu (2023). A resource-efficient ECG diagnosis model for mobile health devices. Information Sciences doi:10.1016/j.ins.2023.119628
- **S33** Junxin Chen, Bo Fang, Haoyue Li et al. (2024). EMCNet: Ensemble Multiscale Convolutional Neural Network for Single-Lead ECG Classification in Wearable Devices. IEEE Sensors Journal doi:10.1109/jsen.2024.3358997
- **S34** J Rosengarten, E Hart, M Ojo et al. (2023). AI remote ECG monitoring improves arrhythmia detection. European Heart Journal doi:10.1093/eurheartj/ehad655.2966
- **S35** Sunghan Lee, Guangyao Zheng, Jeonghwan Koh et al. (2025). Optimizing beat-wise input for arrhythmia detection using 1-D convolutional neural networks: A real-world ECG study. Computer methods and programs in biomedicine doi:10.1016/j.cmpb.2025.108898
- **S36** Ali Rida Ismail, Slavisa Jovanovic, Naeem Ramzan et al. (2023). ECG Classification Using an Optimal Temporal Convolutional Network for Remote Health Monitoring. Sensors (Basel, Switzerland) doi:10.3390/s23031697
- **S37** Alireza Sadeghi, Farshid Hajati, Alireza Rezaee et al. (2024). 3DECG-Net: ECG fusion network for multi-label cardiac arrhythmia detection. Computers in biology and medicine doi:10.1016/j.compbiomed.2024.109126
- **S38** Aayush Panwar, Modigari Narendra, Arnav Arya et al. (2025). Integrated portable ECG monitoring system with CNN classification for early arrhythmia detection. Frontiers in digital health doi:10.3389/fdgth.2025.1535335
- **S39** Jiawen Deng, Jie Yang, Xin'an Wang et al. (2024). A Novel Instruction Driven 1-D CNN Processor for ECG Classification. Sensors (Basel, Switzerland) doi:10.3390/s24134376
- **S40** Md Alamin Talukder, Amira Samy Talaat, Nusrat Jahan Muna et al. (2025). An explainable deep learning framework for trustworthy arrhythmia detection from ECG signals. Scientific reports doi:10.1038/s41598-025-22986-0
- **S41** Moirangthem Tiken Singh, Manibhushan Yaikhom, Rabinder Kumar Prasad (2026). Hybrid feature engineering for resource-efficient Arrhythmia detection in electrocardiogram signals: An interpretable, separability-driven framework. Computers in biology and medicine doi:10.1016/j.compbiomed.2026.111933
- **S42** Ananya Mantravadi, Siddharth Saini, Sai Chandra Teja R et al. (2024). CLINet: A novel deep learning network for ECG signal classification. Journal of electrocardiology doi:10.1016/j.jelectrocard.2024.01.004
- **S43** Muhammad Imran Khalid, Altaf Hussain, Nasir Hussain et al. (2026). Lightweight and interpretable edge intelligence AI with intrusion detection for trustworthy cardiac arrhythmia in medical IoT. Scientific reports doi:10.1038/s41598-026-43578-6
- **S44** Yicheng Li, Yong Xia (2023). Atrial fibrillation detection with signal decomposition and dilated residual neural network. Physiological measurement doi:10.1088/1361-6579/acfa61
- **S45** Mengfan Wu, Ziqi Chen, Niannian Yu et al. (2025). Charge-Trap Memory with Engineered Temporal Dynamics for Physically Integrated Reservoir Computing. Small science doi:10.1002/smsc.202500356
- **S46** Tamás Dózsa, Federico Deuschle, Bram Cornelis et al. (2024). Variable Projection Support Vector Machines and Some Applications Using Adaptive Hermite Expansions. International journal of neural systems doi:10.1142/s0129065724500047
- **S47** Antonino Faraone, Halla Sigurthorsdottir, Ricard Delgado-Gonzalo (2021). Atrial Fibrillation Detection on Low-Power Wearables using Knowledge Distillation. Annual International Conference of the IEEE Engineering in Medicine and Biology Society. IEEE Engineering in Medicine and Biology Society. Annual International Conference doi:10.1109/embc46164.2021.9630957
- **S48** Xinlong Yang, Tianming Cai, Junbin Zang et al. (2025). A geometric insight into ECG classification: leveraging phase space reconstruction and distance to measure. Computer methods in biomechanics and biomedical engineering doi:10.1080/10255842.2025.2582763
- **S49** Tiancheng Cao, Hongtao Li, Hen-Wei Huang (2025). LiMO: A Lightweight MambaOut system for end-to-end IoMT ECG Diagnosis with Fully Configurable Quantization Co-Design on MCU. Annual International Conference of the IEEE Engineering in Medicine and Biology Society. IEEE Engineering in Medicine and Biology Society. Annual International Conference doi:10.1109/embc58623.2025.11253945
- **S50** Yupeng Qiang, Xunde Dong, Xiuling Liu et al. (2024). Conv-RGNN: An efficient Convolutional Residual Graph Neural Network for ECG classification. Computer methods and programs in biomedicine doi:10.1016/j.cmpb.2024.108406
- **S51** Ukesh Thapa, Bipun Man Pati, Attaphongse Taparugssanagorn et al. (2025). Generalizable Hybrid Wavelet-Deep Learning Architecture for Robust Arrhythmia Detection in Wearable ECG Monitoring. Sensors (Basel, Switzerland) doi:10.3390/s25216590
- **S52** Ishan Sahu, Arijit Ukil, Sundeep Khandelwal et al. (2022). LTH-ECG: Lottery Ticket Hypothesis-based Deep Learning Model Compression for Atrial Fibrillation Detection from Single Lead ECG On Wearable and Implantable Devices. Annual International Conference of the IEEE Engineering in Medicine and Biology Society. IEEE Engineering in Medicine and Biology Society. Annual International Conference doi:10.1109/embc48229.2022.9871259
- **S53** Haotian Tang, Mingke Yan, Xidong Wu et al. (2026). Lightweight element-wise product enhanced neural network for efficient arrhythmia detection on embedded devices. Computer methods and programs in biomedicine doi:10.1016/j.cmpb.2025.109101
- **S54** Kartikeya M Menon, Subrat Das, Mark Shervey et al. (2023). Automated electrocardiogram signal quality assessment based on Fourier analysis and template matching. Journal of clinical monitoring and computing doi:10.1007/s10877-022-00948-5
- **S55** Samin Yaser, Muhammad E H Chowdhury, Rusab Sarmun (2025). SER inspired deep learning approach to detect cardiac arrhythmias in electrocardiogram signals using Temporal Convolutional Network and graph neural network. Computers in biology and medicine doi:10.1016/j.compbiomed.2025.110536
