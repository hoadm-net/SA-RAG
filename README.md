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
Each unit is prefixed with a breadcrumb: `<document title> > Điều <n>. <article title> > Khoản <m>`.

### C2. Hierarchical retrieval
Clauses are retrieved with BM25 (on word-segmented text) and bge-m3 dense retrieval, and the two rankings are fused
with Reciprocal Rank Fusion (RRF). Clause scores are then aggregated into article scores (max by default; sum of
top-2 as a variant), since gold labels in all datasets are at the article level (or document level).

### C3. Structural expansion
- Parent lead-in: each selected clause is accompanied by its article title and the introductory sentence that
  precedes clause 1.
- Cross-reference resolution: references such as `khoản \d+ Điều \d+` or `Điều \d+ của (Luật|Nghị định) này` are
  resolved within the same document, and the referenced clause or article is added (one hop).

### C4. Budget-aware packing and citation-grounded generation
- For a token budget B in {1024, 2048, 4096}, counted with the generator's tokenizer, units are selected greedily by
  score per token until the budget is reached. Duplicates are removed and units from the same article keep their
  original order.
- The generator is prompted in Vietnamese to answer with citations of the form `[<doc_id>, Điều X, Khoản Y]`.
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
| Upper bound | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | Qwen2.5 Technical Report |
| Dense retriever | bge-m3 | `BAAI/bge-m3` | M3-Embedding |
| Sparse retriever | BM25 (`bm25s`) | - | Robertson & Zaragoza |
| Word segmentation | pyvi / underthesea | - | - |
| Judge | GPT or Claude, via OpenRouter | see `configs/models.yaml` | Provider technical report |

## 5. Experiments

### 5.1 Baselines
All systems share the same retrievers, generator and token budget.
- B1, fixed chunking: 256-token sliding windows (overlap 32), hybrid retrieval, top-k truncated to the budget.
- B2, article chunking: one chunk per article, hybrid retrieval, top-k truncated to the budget.
- SA-RAG: C1 + C2 + C3 + C4.

### 5.2 Ablations
Run on SA-RAG with B = 2048 and Qwen2.5-3B-Instruct:
- without breadcrumbs (C1)
- without clause-to-article aggregation, ranking clauses directly (C2)
- without structural expansion (C3)
- without budget-aware packing, using plain top-k truncation (C4)
- without citation verification (C4)

### 5.3 Budget sweep
B in {1024, 2048, 4096}, for every generator and for B1, B2 and SA-RAG.

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
- [ ] Clause units with breadcrumbs; BM25 and bge-m3 indexes
- [ ] RRF and article aggregation; retrieval results for B1, B2, SA-RAG
- [ ] Structural expansion (lead-in, cross-references)
- [ ] Budget-aware packing
- [ ] Generation, citation parsing and verification
- [ ] Main experiments, ablations, budget sweep
- [ ] Judge evaluation and bootstrap confidence intervals
- [ ] Local llama.cpp benchmarks
- [ ] Tables and figures

## 9. References

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
