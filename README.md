# Structure-aware Retrieval-Augmented Generation for Vietnamese Legal Question Answering with Small Language Models

## 1. Overview

Small language models (SLMs, 1-3B parameters) can run locally on commodity hardware, but their effective context is
narrow and their answers degrade quickly when the retrieved text is long or noisy. Vietnamese legal documents follow
an explicit hierarchy: Văn bản (document), Chương (chapter), Điều (article), Khoản (clause), Điểm (point).

SA-RAG uses this hierarchy to give an SLM correct and sufficient evidence under a tight token budget, and to make
its answers traceable to specific articles and clauses.

Scope:
- Training-free: no fine-tuning or adapters; only off-the-shelf retrievers and LLMs.
- Public datasets only; no new dataset or benchmark is built.
- Every model and tool used has a published paper or technical report.

## 2. Method

SA-RAG has four components. Each can be switched off independently for ablation.

### C1. Structure-aware indexing
Each legal document is parsed into a tree (document / chapter / article / clause / point) using regular expressions:

| Level | Pattern |
|---|---|
| Chapter | `^Chương\s+[IVXLC]+` |
| Article | `^Điều\s+\d+[a-z]?\.` |
| Clause | `^\d+\.\s` |
| Point | `^[a-zđ]\)\s` |

Two input forms are supported. Full-text documents (ViLegalExperts) are parsed with all four patterns. Corpora that
are already split into articles (ALQAC, Zalo) skip the chapter and article levels: the first line of each article is
its title, the text before clause 1 is the lead-in, and clauses and points are parsed inside the article.

Parsing rules:
- Clauses must be numbered 1, 2, 3, ... and points follow the Vietnamese alphabet (a, b, c, d, đ, e, g, ...)
  within their parent. A numbered line that breaks the sequence (a year, an amount, a quoted amendment) is kept as
  continuation text and counted.
- Articles must appear in increasing order, which ignores article headings quoted inside amendments. Numbering may
  restart at Điều 1 only after a signature block or an "ban hành kèm theo" heading, which starts a new part (for
  example, a decision followed by the regulation it issues, or annexed forms). Lookups use the main part.
- Soft line wraps in scraped text are joined, trailer tags (`Chương: II`, `Mục: 3`) are attached to their article,
  and signature blocks, annexes and enactment formulas after the last article are removed.
- A document with no article becomes one fallback unit; the fallback rate is reported.

Parse results (`scripts/02_parse_corpus.py`):

| Corpus | Documents | Articles | Clauses | Clean articles | Fallback documents | Gold references resolved |
|---|---|---|---|---|---|---|
| ALQAC 2024 | 17 | 2,249 | 6,356 | 99.6% | 0% | 100% (article) |
| ALQAC 2025 | 28 | 3,363 | 11,123 | 99.6% | 0% | 100% (article) |
| Zalo 2021 | 3,271 | 61,425 | 195,594 | 98.3% | 0% | 100% (article) |
| ViLegalExperts | 9,996 | 56,636 | 123,295 | 96.5% | 22.8% | 96.1% (article), 88.5% (clause), silver |

A clean article has no out-of-sequence numbered line. ViLegalExperts fallback documents are decisions and official
letters without articles. Most of its unresolved silver references point to another document (for example, an
amending decree) or to articles missing from the released text.

The index unit is the clause, together with its points. An article with no clauses is indexed as a single unit.
Each unit is prefixed with a breadcrumb: `<document title> > Điều <n>. <article title> > Khoản <m>`. The article
lead-in is not part of a clause unit; C3 adds it at packing time. Units longer than 512 bge-m3 tokens are split on
line boundaries (sentence or word boundaries for a single long line), and each piece keeps the breadcrumb and its
article mapping. Document codes are made readable in breadcrumbs (`47/2011/tt-bca` becomes `Thông tư 47/2011/TT-BCA`).

Four index variants are built per corpus (`scripts/03_build_index.py`): clause units with breadcrumbs (SA-RAG),
clause units without breadcrumbs (ablation), whole articles (B2, split only beyond 4,096 tokens) and 256-token
windows with overlap 32 (B1). Every unit records the article(s) it covers, so all variants are scored at the article
level.

### C2. Hierarchical retrieval
Clauses are retrieved with BM25 (on word-segmented text) and bge-m3 dense retrieval, and the two rankings are fused
with weighted Reciprocal Rank Fusion (RRF, k = 60; each retriever returns 200 candidates). The BM25 weight (0.5,
dense 1.0) was chosen on the ALQAC 2024 dev set only; with equal weights BM25 pulled the fused ranking below
dense-only retrieval on Zalo. Clause scores are then aggregated into article scores (max by default; sum of
top-2 as a variant), since gold labels in all datasets are at the article level (or document level). With rank-based
RRF scores, max aggregation ranks articles exactly as their best clause, so the "without aggregation" ablation only
differs downstream (packing, C4); sum of top-2 favours long articles and lowers R@1 (0.835 to 0.619 on ALQAC 2024).

### C3. Structural expansion
- Parent lead-in: each selected clause is accompanied by its article title and the introductory sentence that
  precedes clause 1.
- Cross-reference resolution: references such as `khoản \d+ Điều \d+` or `Điều \d+ của (Luật|Nghị định) này` are
  resolved within the same document, and the referenced clause or article is added (one hop).

### C4. Budget-aware packing and citation-grounded generation
- For a token budget B in {512, 1024, 2048, 4096}, counted with the generator's tokenizer over the whole chat-formatted
  prompt, units are added greedily in priority order, skipping any unit that would overflow, until nothing else
  fits. The budget is checked on the rendered context, so article headers and lead-ins are paid for. Duplicates are
  removed and units from the same article keep their original order.
- Priority blends a unit's own score with its article's aggregated score (weight 0.5); without aggregation the
  unit score is used alone. Dividing by the unit's length (score per token) was tested on the dev set and lowered
  context recall from 0.98 to 0.80 at B = 1024, because RRF scores are flat across ranks while clause lengths vary
  several-fold, so the length exponent is 0.
- The context is grouped by article: `[<document>, Điều X. <title>]`, the lead-in (C3), then the clauses.
- The generator is prompted in Vietnamese to answer with citations of the form `[<document>, Điều X, Khoản Y]`.
  The prompt shows the format with placeholders only: on the dev set a concrete example citation was copied
  verbatim by the 1.5B model and a concrete example answer biased true/false predictions.
- Citations in the output are parsed and checked against the document tree and the packed context; invalid
  citations are removed and counted.

## 3. Datasets

| Dataset | Use | Test set | Corpus | Gold labels |
|---|---|---|---|---|
| ALQAC 2024 (main) | Retrieval, QA | public + private test, 308 questions (true/false, multiple choice, short free text) | 17 laws, 2,249 articles, pre-split by article | article |
| ALQAC 2025 | Retrieval, QA (second test set) | private test, 82 questions (mostly scenario-based multiple choice) | 28 laws, 3,363 articles, pre-split by article | article |
| Zalo Legal Text Retrieval 2021 | Retrieval at scale | 3,196 citizen questions | 3,271 documents (laws, decrees, circulars), 61,425 articles | article |
| ViLegalExperts (Nguyen et al., 2026) | Long-form QA, parsing of full-text documents | 773 test questions whose source document is released | 9,996 full-text documents with chapter / article / clause / point structure | document; article references extracted from the expert answer (silver) |

- ALQAC 2024, ALQAC 2025 and Zalo come from Hugging Face `nguyenlab/ALQAC` (gated). ViLegalExperts comes from
  `DatNguyenAIML/ViLegalExperts` (arXiv:2609.39189).
- Dev set: the 100 expert-verified ALQAC 2024 training questions, used only for prompt design.
- Zalo has no answers and is used for retrieval only. No official labelled test split is public, and the method
  is training-free, so all 3,196 questions are evaluated.
- ViLegalExperts: only 773 of the 17,821 test questions currently have their source document in the release; we
  evaluate on these, with all 9,996 released documents as the retrieval corpus. Article-level labels are silver
  labels parsed from the expert answers (`Điều N`, `khoản M Điều N`) and are reported separately from gold metrics.
- VLQA (Nguyen et al., 2025) was planned but is not publicly released; it will be added if it becomes available.
- Some ALQAC article ids are absent from the corpus (Luật Trồng trọt 45, 64; Hiến pháp 115, 116; Luật Trọng tài
  thương mại 68). Cross-references to them are counted as unresolvable.

All reported results are on the test splits. Retrieval is evaluated at the article level (document level as well
for ViLegalExperts).

## 4. Models

All generation uses greedy decoding (temperature 0).

| Role | Model | Hugging Face id | Reference |
|---|---|---|---|
| Generator | Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | Qwen2.5 Technical Report |
| Generator | Qwen2.5-3B-Instruct | `Qwen/Qwen2.5-3B-Instruct` | Qwen2.5 Technical Report |
| Generator | Qwen3-1.7B (non-thinking mode) | `Qwen/Qwen3-1.7B` | Qwen3 Technical Report |
| Generator | Sailor2-1B-Chat | `sail/Sailor2-1B-Chat` | Sailor2 |
| Upper bound | Qwen2.5-7B-Instruct (FP8 weights in vLLM, to fit a 16 GB GPU) | `Qwen/Qwen2.5-7B-Instruct` | Qwen2.5 Technical Report |
| Dense retriever | bge-m3 | `BAAI/bge-m3` | M3-Embedding |
| Sparse retriever | BM25 (`bm25s`) | - | Robertson & Zaragoza |
| Word segmentation | pyvi / underthesea | - | - |
| Judge | Claude Haiku 5.5, via OpenRouter (`anthropic/claude-haiku-5.5`) | - | Provider model card |

## 5. Experiments

### 5.1 Baselines
All systems share the same retrievers, generator and token budget.
- B1, fixed chunking: 256-token sliding windows (overlap 32), hybrid retrieval, top-k truncated to the budget.
- B2, article chunking: one chunk per article, hybrid retrieval, top-k truncated to the budget.
- SA-RAG: C1 + C2 + C3 + C4.

### 5.2 Ablations
Run on SA-RAG with B = 2048 and Qwen2.5-3B-Instruct:
- without breadcrumbs (C1)
- without clause-to-article aggregation (C2): packing priority uses each clause's own score only (article weight 0);
  with max aggregation the article ranking itself equals the ranking by best clause, so retrieval metrics are unchanged
- without structural expansion (C3)
- without budget-aware packing, using plain top-k truncation (C4)
- without citation verification (C4)

### 5.3 Budget sweep
B in {512, 1024, 2048, 4096}, for every generator and for B1, B2 and SA-RAG.

### 5.4 Local deployment
Generators are converted to GGUF Q4_K_M and run with llama.cpp on a commodity machine (laptop CPU or a 6-8 GB GPU).
We report time to first token, tokens per second, end-to-end latency per question, and peak RAM/VRAM.

### 5.5 Metrics
- Retrieval (article level): Recall@{1, 3, 5, 10}, MRR@10.
- QA, ALQAC: accuracy on true/false and multiple-choice questions; token F1 and judge score on free-text questions.
- QA, ViLegalExperts: judge correctness (1-5 scale and binary) and token F1 against the expert answer.
- Citation: precision and recall against gold articles; share of citations that exist in the document tree.
- Means over the test set, with bootstrap 95% confidence intervals for the main results.

### 5.6 Result tables and figures
- Table 1: retrieval, QA and citation results for B1, B2 and SA-RAG with all generators at B = 2048.
- Table 2: ablations.
- Figure 1: quality versus budget, annotated with local latency and memory.
- Pipeline diagram of the method.

## 6. Repository structure

```
SA-RAG/
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml           # paths, budgets, k, seeds
│   ├── models.yaml            # generator, retriever and judge ids
│   └── experiments/           # one file per experiment (baselines, ablations, sweep)
├── data/
│   ├── raw/                   # downloaded datasets (not tracked)
│   └── processed/             # parsed trees and index units (not tracked)
├── src/sarag/
│   ├── parsing/               # legal document to tree (C1)
│   ├── indexing/              # BM25 and bge-m3 indexes, breadcrumbs (C1)
│   ├── retrieval/             # hybrid retrieval, RRF, aggregation (C2)
│   ├── expansion/             # parent lead-in, cross-references (C3)
│   ├── packing/               # budget-aware packing (C4)
│   ├── generation/            # prompts, vLLM and llama.cpp backends, citation checking (C4)
│   ├── baselines/             # B1 fixed chunking, B2 article chunking
│   ├── evaluation/            # retrieval, QA and citation metrics, judge
│   └── utils/
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_parse_corpus.py
│   ├── 03_build_index.py
│   ├── 04_run_retrieval.py
│   ├── 05_run_generation.py
│   ├── 06_evaluate.py
│   ├── 07_benchmark_local.py  # llama.cpp latency and memory
│   └── 08_make_tables_figures.py
├── outputs/                   # predictions, metrics, tables, figures
└── tests/                     # parser, cross-reference patterns, packing, citation parser
```

## 7. Implementation details

- Python 3.10+. Prompts are in Vietnamese.
- vLLM is used for the main experiments and llama.cpp (`llama-cpp-python`) for local benchmarks.
- The judge is called through OpenRouter's OpenAI-compatible API (`OPENROUTER_API_KEY`). A single model version is
  used for all runs at temperature 0, and every response is cached by prompt hash.
- All hyperparameters live in YAML configs. Each prediction is stored as a JSONL record with `question_id`,
  `config`, `retrieved_ids`, `context_ids`, `answer`, `citations` and `latency`.
- Embeddings and retrieval results are cached; generation can resume from partial outputs.
- Token budgets are measured with the generator's tokenizer and include the prompt template.
- Documents that cannot be parsed into a tree fall back to article-level units; the fallback rate is reported.
- Unit tests cover the parser, the cross-reference patterns, the packing budget limit and the citation parser.

## 8. Roadmap

- [x] Data: download and statistics for ALQAC 2024/2025, Zalo 2021, ViLegalExperts; VLQA not released
- [x] Parser and unit tests; parse success rate
- [x] Clause units with breadcrumbs; BM25 and bge-m3 indexes
- [x] RRF and article aggregation; retrieval results for B1, B2, SA-RAG
- [x] Structural expansion (lead-in, cross-references)
- [x] Budget-aware packing
- [x] Generation, citation parsing and verification
- [x] Main experiments, ablations, budget sweep (B in {512, 1024, 2048, 4096})
- [x] Bootstrap confidence intervals
- [x] Judge evaluation (Claude Haiku 5.5)
- [x] Local llama.cpp benchmarks
- [x] Tables and figures

## 9. Results

Full tables: `outputs/tables/tables.md` (built by `scripts/08_make_tables_figures.py`); Figure 1:
`outputs/figures/figure1_budget.png`. Means over the test sets; 95% CIs by bootstrap.

### 9.1 Retrieval

Article-level retrieval on the test sets (hybrid retrieval, final settings; ViLegalExperts uses silver labels):

| System | ALQAC 2024 R@1 / MRR@10 | ALQAC 2025 R@1 / MRR@10 | Zalo R@1 / MRR@10 | ViLegalExperts R@1 / MRR@10 |
|---|---|---|---|---|
| B1, fixed 256-token windows | 0.496 / 0.678 | 0.518 / 0.668 | 0.403 / 0.555 | 0.398 / 0.610 |
| B2, article chunks | 0.781 / 0.862 | 0.713 / 0.804 | 0.501 / 0.635 | 0.542 / 0.741 |
| SA-RAG, clause units + breadcrumbs | **0.841 / 0.897** | **0.726 / 0.817** | **0.539 / 0.660** | **0.580 / 0.762** |
| SA-RAG without breadcrumbs | 0.772 / 0.833 | 0.665 / 0.755 | 0.524 / 0.650 | 0.471 / 0.650 |

Breadcrumbs matter most where clause text alone lacks context (ViLegalExperts full texts, +0.109 R@1) and least on
Zalo, whose articles have no titles and whose document names are codes.

Context recall (share of gold articles inside the packed context, Qwen2.5 tokenizer) is highest for SA-RAG at
B <= 1024 on every dataset, e.g. at B = 1024: ALQAC 2025 0.927 vs 0.805 (B2), ViLegalExperts 0.739 vs 0.669 (B2).
At B = 4096 the three systems are within 0.03 of each other.

### 9.2 Answers (paired, SA-RAG minus baseline, pooled over generators, CI by bootstrap over questions)

| Baseline | Data, metric | B = 512 | B = 1024 | B = 2048 | B = 4096 |
|---|---|---|---|---|---|
| B2 | ALQAC accuracy (4 Qwen models, 370 questions) | **+0.026** | +0.007 | -0.003 | -0.007 |
| B2 | ViLegalExperts token F1 (5 models, 773 questions) | +0.004 | **+0.022** | **+0.017** | **+0.007** |
| B2 | ViLegalExperts judge score, 1-5 | **+0.113** | **+0.105** | +0.024 | -0.039 |
| B1 | ALQAC accuracy | **+0.049** | +0.002 | +0.007 | -0.009 |
| B1 | ViLegalExperts token F1 | **+0.033** | **+0.039** | **+0.029** | **+0.025** |
| B1 | ViLegalExperts judge score | **+0.278** | **+0.315** | **+0.165** | **+0.137** |

Bold: CI excludes 0. SA-RAG helps most under tight budgets and the advantage fades at B = 4096, where all systems
fit most of the gold evidence. ALQAC accuracy is near its ceiling once the gold article is in context. Citation
precision and recall of SA-RAG and B2 are close (within about 0.05 either way), and both are clearly above B1.

### 9.3 Ablations (Qwen2.5-3B at B = 1024 and 2048, Qwen3-1.7B and Qwen2.5-7B at B = 2048)

| Variant | Accuracy, mean change | Long-form F1, mean change | Citation precision, mean change |
|---|---|---|---|
| without breadcrumbs (C1) | -0.025 (worse in 88% of settings) | -0.020 (100%) | -0.068 (100%) |
| without aggregation (C2) | +0.001 | -0.004 | +0.008 |
| without structural expansion (C3) | -0.005 | -0.003 | +0.003 |
| without budget-aware packing (C4) | +0.003 | -0.005 | -0.003 |
| without citation verification (C4) | +0.001 | 0.000 | -0.079 (100%) |

Breadcrumbs and citation verification have consistent effects. Aggregation, structural expansion and the packing
strategy show no effect beyond run-to-run variation: repeating identical runs with vLLM changed accuracy by at most
one question and citation precision by up to 0.03.

### 9.4 Local deployment (llama.cpp, GGUF Q4_K_M, mean per question at B = 2048)

| Generator | GPU: TTFT / latency / VRAM | CPU, 8 threads: TTFT / tokens per s / latency / RAM |
|---|---|---|
| Sailor2-1B | 0.12 s / 0.58 s / 1.1 GB | 0.37 s / 55 / 3.7 s / 2.1 GB |
| Qwen2.5-1.5B | 0.09 s / 0.37 s / 1.7 GB | 0.40 s / 43 / 2.9 s / 2.2 GB |
| Qwen3-1.7B | 0.11 s / 0.42 s / 2.0 GB | 0.53 s / 33 / 4.1 s / 2.9 GB |
| Qwen2.5-3B | 0.16 s / 0.52 s / 2.4 GB | 0.74 s / 23 / 4.5 s / 3.2 GB |
| Qwen2.5-7B | 0.30 s / 0.85 s / 4.8 GB | 1.49 s / 11 / 9.0 s / 5.9 GB |

Every generator fits a 6 GB GPU. The CPU figures come from 8 threads of a server CPU (a laptop will be slower), and
llama.cpp reuses the KV cache of the shared system prompt, which lowers TTFT. Numbers for all budgets are in
`outputs/metrics/local_benchmark.jsonl`.

### 9.5 Caveats

- Sailor2-1B does not follow the answer format on ALQAC (it answers "Đúng" to every true/false question and often
  copies the answer placeholder on multiple choice), so it is excluded from ALQAC accuracy and kept for
  ViLegalExperts. Its 4k context window leaves no room for an answer at B = 4096, which is not run.
- Qwen2.5-7B runs with FP8 weights in vLLM to fit a 16 GB GPU.
- ViLegalExperts article labels are silver (parsed from expert answers); its judge is Claude Haiku 5.5 with
  reasoning disabled, one fixed version, temperature 0, responses cached by prompt hash.
- Citations are re-parsed from the raw outputs at evaluation time against each run's packed context, so the
  reported citation metrics always use the current parser.

## 10. References

- Qwen Team. Qwen2.5 Technical Report. 2024.
- Qwen Team. Qwen3 Technical Report. 2025.
- Sailor2 Team. Sailor2: Sailing in South-East Asia with Inclusive Multilingual LLMs. 2025.
- Chen et al. M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. 2024.
- Nguyen et al. VLQA: The First Comprehensive, Large, and High-Quality Vietnamese Dataset for Legal Question Answering. arXiv:2507.19995, 2025.
- ALQAC 2024 and 2025: Automated Legal Question Answering Competition.
- Zalo AI Challenge 2021: Legal Text Retrieval.
- Nguyen et al. ViLegalExpert: A Large-Scale Benchmark for Vietnamese Legal Retrieval and Question Answering from Real-World Consultations. arXiv:2609.39189, 2026.
- Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.
- Cormack et al. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR 2009.
- Robertson and Zaragoza. The Probabilistic Relevance Framework: BM25 and Beyond. 2009.
