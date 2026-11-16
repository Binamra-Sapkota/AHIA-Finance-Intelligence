# AHIA-R2 Book Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch Pure](https://img.shields.io/badge/pytorch-pure%20implementation-red.svg)](https://pytorch.org/)
[![UI](https://img.shields.io/badge/gui-PySide6-green.svg)](https://wiki.qt.io/Qt_for_Python)
[![Storage](https://img.shields.io/badge/storage-SQLite-lightgrey.svg)](https://sqlite.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Zero Pretrained Weights](https://img.shields.io/badge/weights-random%20init%20only-orange.svg)](#non-negotiable-operational-rules)

**AHIA-R2 Book Intelligence** is an open-source, fully offline desktop AI research demonstration. The system couples classical **Okapi BM25** Information Retrieval with a custom, compact **decoder-only Transformer** trained entirely from random initialization (no pretrained weights, no pretrained embeddings, zero remote AI APIs). It is grounded on a curated local library of six foundational personal-finance books.

---

## 1. Architectural Highlights

- **Pure PyTorch Decoder-Only Transformer**:
  - **Blocks**: 8 Transformer layers
  - **Hidden Dimension ($d_{\text{model}}$)**: 256
  - **Attention Heads**: 4 heads ($d_k = 64$)
  - **Feed-Forward Dimension ($d_{\text{ff}}$)**: 1024
  - **Vocabulary Target**: 8,000 tokens (Byte-Level BPE trained from scratch)
  - **Maximum Context ($L_{\text{ctx}}$)**: 512 tokens
  - **Adaptive Execution Depths**: Variable exit depths at 2, 4, and 8 blocks for dynamic compute allocation
  - **MoE Roadmap**: Architectural hooks for 4 planned Mixture-of-Experts, scheduled after dense baseline stabilization
- **Grounding & Retrieval Engine**:
  - Okapi BM25 / BM25+ inverted index with parameter tuning ($k_1 = 1.5, b = 0.75$)
  - Persistent SQLite storage for book metadata, chapter hierarchies, chunk spans, and citation caches
- **Bounded Adaptive Controller**:
  - Deterministic reasoning loop ceiling ($T_{\max} \le 3$ iterations)
  - Query reformulation and confidence thresholding
- **Independent Resource Governor**:
  - External supervisor watchdog enforcing a hard 10.0s CPU wall-clock ceiling
  - Memory boundary monitoring and hard generation token limits (256 tokens)
- **PySide6 Desktop Application**:
  - 100% offline, native desktop interface
  - Asynchronous thread-pool execution ensuring non-blocking UI
  - Total decoupling: the UI never imports PyTorch; interactions proceed via typed Data Transfer Objects (DTOs)

---

## 2. The 10 Decoupled Subsystems

The system architecture cleanly isolates ten modular subsystems:

| # | Subsystem | Module | Core Functionality |
|---|---|---|---|
| 1 | **Desktop Interface** | `ui/` | PySide6 GUI: chat canvas, citation inspector, library manager, and diagnostics |
| 2 | **Book Ingestion** | `core/ingestion/` | Multi-format parsers (TXT, PDF, EPUB), normalizer, sliding-window chunker |
| 3 | **Tokenizer & Datasets** | `core/tokenizer/`, `core/dataset/` | Custom Byte-Level BPE tokenizer (8,000 vocab) and RAG synthesis dataset pipelines |
| 4 | **Neural Model** | `core/model/` | 8-block decoder-only Transformer with KV-cache, RoPE, RMSNorm, and adaptive depth exits (2, 4, 8) |
| 5 | **Training Engine** | `core/training/` | Pure PyTorch causal LM and context-conditioned synthesis trainer (random init) |
| 6 | **Adaptive Controller** | `core/controller/` | Bounded reasoning loop ($T_{\max} \le 3$), query decomposition, depth selection |
| 7 | **Knowledge Retrieval** | `core/retrieval/` | Pure-Python Okapi BM25 indexer and search engine backed by SQLite |
| 8 | **Reliability Engine** | `core/reliability/` | Verbatim quote validator, citation ledger, hallucination suppression, confidence scoring |
| 9 | **Resource Governor** | `core/governor/` | Independent supervisor thread enforcing wall-clock timeout and memory bounds |
| 10 | **Evaluation Suite** | `core/evaluation/` | Retrieval recall@K, citation fidelity, deterministic bound tests, latency benchmarks |

---

## 3. Non-Negotiable Operational Rules

1. **No Pretrained Language-Model Weights**: Model is initialized strictly from random initialization. Pretrained weights are forbidden.
2. **No External AI APIs in Runtime**: Zero runtime network access. Runs 100% locally.
3. **No Pretrained Embeddings**: Semantic retrieval is driven by BM25; token embeddings are learned from scratch.
4. **Custom Tokenizer from Scratch**: Byte-Level BPE trained exclusively on authorized domain text.
5. **Pure PyTorch Implementation**: All neural layers, attention, and loss functions written in PyTorch.
6. **BM25 for Initial Retrieval**: Classical Okapi BM25 for candidate passage selection.
7. **PySide6 Desktop App**: Native desktop UI built with PySide6.
8. **100% Offline Usability**: Fully operational without internet connectivity post-setup.
9. **Zero Exaggeration of Intelligence**: Transparently presented as a bounded, compact research demonstration.
10. **Strict Factual Grounding**: Zero fabrication of citations or passages. Missing context yields graceful refusal.
11. **Strict Copyright Protection**: Copyrighted texts and raw books are never redistributed or committed to Git.
12. **Git Repository Hygiene**: Private files (`data/raw/`, `checkpoints/`, `data/processed/`) are protected by `.gitignore`.
13. **Full Test Coverage for Adaptive Paths**: All adaptive branches (reformulation, early exit, fallback) are tested.
14. **Deterministic Reasoning Limits**: Hard cap $T_{\max} = 3$ iterations on reasoning loops.
15. **Independent Resource Enforcement**: Hard timeout and memory caps enforced by an external supervisor.

---

## 4. Documentation Index

- **[System Architecture](docs/architecture.md)**: Exhaustive architectural blueprint, component interfaces, mathematical formulations, and data flow.
- **[Training Plan](docs/training_plan.md)**: Pretraining & grounded synthesis fine-tuning strategy, loss formulations, and compute budgets.
- **[Data Policy & Safe Harbor](docs/data_policy.md)**: Copyright compliance, local storage isolation, and dataset ethics.
- **[Development Roadmap](docs/development_roadmap.md)**: Phased milestones (Phases 0 through 9) and exit criteria.
- **[Operational Governance (AGENTS.md)](AGENTS.md)**: Engineering standards and developer protocols.

---

## 5. Quickstart & Environment Setup

### 5.1 Prerequisites
- Linux / Windows / macOS
- Python 3.10+
- Consumer CPU (e.g. AMD Ryzen 5 / Intel i5/i7) with $\ge 4$ GB available RAM

### 5.2 Setup Environment
```bash
# Clone the repository
git clone https://github.com/example/AHIA-Finance-Intelligence.git
cd AHIA-Finance-Intelligence

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 5.3 Verify Repository Consistency
```bash
pytest tests/test_repository_consistency.py
```

---

## 6. License & Disclaimer

- **License**: Apache 2.0 (see [LICENSE](LICENSE)).
- **Disclaimer**: AHIA-R2 is a specialized research demonstration with bounded domain capabilities. It does not provide financial or investment advice.
