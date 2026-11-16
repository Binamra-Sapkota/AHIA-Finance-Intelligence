# AGENTS.md - Engineering Governance & Operational Protocol
## Project: AHIA-R2 Book Intelligence

---

### 1. Mission & Architecture Identity

**AHIA-R2 Book Intelligence** is an open-source research demonstration that builds a compact, self-contained desktop AI application powered by a small decoder-only Transformer trained entirely from random initialization (no pretrained weights, no external LLM APIs, no pretrained embeddings). The system indexes and grounds its knowledge upon a curated local library of six personal-finance books using classical Okapi BM25 retrieval, an adaptive bounded reasoning controller, an independent resource governor, persistent SQLite cataloging, and a PySide6 desktop interface.

All contributing human engineers and AI agents must strictly adhere to the policies, boundaries, and architectural guidelines defined in this document.

---

### 2. The 15 Non-Negotiable Operational Rules

Every code change, design decision, pull request, and operational action is governed by these fifteen rules:

1. **No Pretrained Language-Model Weights**: All neural model weights must be trained from random initialization. Pretrained weights (e.g. HuggingFace GPT-2, LLaMA, BERT, Mistral) are strictly forbidden.
2. **No External AI APIs in Runtime**: The finished desktop application must run fully autonomous on the user's local machine without calling OpenAI, Anthropic, Google, Cohere, or any external AI services.
3. **No Pretrained Embedding Models**: Semantic embeddings cannot come from pretrained models (e.g., sentence-transformers, OpenAI embeddings, fastText). Initial retrieval is driven by BM25, and representation learning is self-contained.
4. **Custom Tokenizer Built from Scratch**: The tokenizer must be trained from scratch on an authorized text corpus using an explicit, inspectable algorithm (Byte-level BPE with 8,000 target vocabulary).
5. **Pure PyTorch Implementation**: All neural network layers, attention mechanisms, loss functions, optimizers, and training loops must be implemented using PyTorch (`torch`, `torch.nn`).
6. **BM25 for Initial Retrieval**: Initial document selection and candidate generation from the book corpus must use Okapi BM25 (or an authorized variant such as BM25+ / BM25L) backed by SQLite storage.
7. **PySide6 for Desktop Application**: The graphical user interface must be developed using PySide6 (Qt for Python).
8. **100% Offline Usability**: After environment setup and local training/indexing, the application must function with zero network access. No telemetry, no remote asset fetching.
9. **Zero Exaggeration of Intelligence**: Never claim commercial-model-level intelligence (e.g. "GPT-4 equivalent"). Explicitly communicate the model's small footprint, bounded domain scope, and architectural nature to users.
10. **Strict Factual Grounding & Zero Fabrication**: Never fabricate quotations, book passages, page numbers, chapter titles, or citations. If a requested concept is missing from the retrieved context, the system must state that the knowledge library does not contain the answer.
11. **Strict Copyright Protection**: Never redistribute copyrighted book texts or unauthorized datasets. Raw books and copyrighted materials must never be committed to source control.
12. **Git Repository Hygiene**: Private book files (`data/raw/`, `data/books/`), processed training corpora (`data/processed/`), indices (`data/indices/`), and model checkpoints (`checkpoints/`, `models/*.pt`) must be kept out of public Git commits via `.gitignore`.
13. **Full Test & Training Coverage for Adaptive Paths**: Every adaptive execution path (e.g., query reformulation, multi-hop context retrieval, adaptive depth exits at 2, 4, 8 blocks, fallback branches) must have explicit unit tests and appropriate training/verification coverage.
14. **Deterministic Reasoning Loop Limits**: All iterative retrieval and reasoning loops must have a strictly enforced deterministic maximum iteration limit ($T_{\max} \le 3$). Infinite loops or unbounded search paths are prohibited.
15. **Independent Resource Limit Enforcement**: System resource limits (CPU timeout ceiling of 10.0s, memory bounds, token generation limits) must be enforced by an independent outer supervisor thread/watchdog, not left solely to the learned neural controller or model stop tokens.

---

### 3. Target Neural Engine Configuration

The core model is a compact decoder-only Transformer with the following exact specifications:
- **Transformer Blocks**: 8 blocks
- **Hidden Dimension ($d_{\text{model}}$)**: 256
- **Attention Heads**: 4 heads ($d_k = 64$)
- **Feed-Forward Dimension ($d_{\text{ff}}$)**: 1024 (SwiGLU or GELU activation)
- **Vocabulary Target**: 8,000 tokens (Byte-Level BPE from scratch)
- **Maximum Context Window ($L_{\text{ctx}}$)**: 512 tokens
- **Adaptive Execution Depths**: Variable forward execution depths at 2, 4, and 8 blocks
- **MoE Roadmap**: 4 planned mixture-of-experts, implemented only after the dense baseline is fully validated

---

### 4. The 10 Subsystem Separation & Ownership Matrix

Architecture mandates total separation between the following ten subsystems:

| Subsystem # | Name | Root Path | Interface Contract | Primary Owner |
|---|---|---|---|---|
| 1 | **Desktop Interface** | `ui/` | `ui/interfaces.py` | Desktop UI Engineer |
| 2 | **Book Ingestion** | `core/ingestion/` | `core/interfaces/ingestion.py` | Data & Retrieval Specialist |
| 3 | **Tokenizer & Datasets** | `core/tokenizer/`, `core/dataset/` | `core/interfaces/tokenizer.py` | Tokenizer & Modeling Specialist |
| 4 | **Neural Model** | `core/model/` | `core/interfaces/model.py` | ML Architect & Lead |
| 5 | **Training Engine** | `core/training/` | `core/interfaces/training.py` | ML Architect & Lead |
| 6 | **Adaptive Controller** | `core/controller/` | `core/interfaces/controller.py` | Reasoning & Safety Controller |
| 7 | **Knowledge Retrieval** | `core/retrieval/` | `core/interfaces/retrieval.py` | Data & Retrieval Specialist |
| 8 | **Reliability Engine** | `core/reliability/` | `core/interfaces/reliability.py` | Reasoning & Safety Controller |
| 9 | **Resource Governor** | `core/governor/` | `core/interfaces/governor.py` | Reasoning & Safety Controller |
| 10 | **Evaluation Suite** | `core/evaluation/` | `core/interfaces/evaluation.py` | Verification & Quality Engineer |

---

### 5. Engineering Standards & Strict Guidelines

1. **Modular Architecture & UI Decoupling**:
   - The UI layer (`ui/`) must **never** import PyTorch (`torch`, `torch.nn`).
   - Communication between the UI and backend passes strictly through typed Data Transfer Objects (`service/dto.py`) and background worker threads (`QRunnable`/`QThread`).
2. **Type Hints**:
   - 100% of public and internal functions, methods, and class attributes must have comprehensive Python type hints (`typing`, `Optional`, `List`, `Dict`, `Tuple`, `Generator`, `Protocol`, `Callable`).
3. **Docstrings**:
   - All modules, classes, and public functions must include Google-style docstrings with explicit Args, Returns, and Raises sections.
4. **Pytest Verification**:
   - Every module must have automated tests in `tests/`.
   - Never claim tests passed without executing them and recording the terminal output.
   - Do not replace failing implementations with silent mock stubs.
5. **No Premature Training**:
   - Do not execute long model training runs until dataset preparation and dense baseline validation phases are formally authorized.
6. **SQLite Storage**:
   - Use SQLite for book cataloging, chapter tree navigation, chunk character spans, and citation ledger verification.

---

### 6. Phased Development Lifecycle Protocol

Every engineering phase follows this 4-step protocol:

1. **Pre-Phase Implementation Plan**:
   - State the specific phase, goals, target files, interfaces, and expected test outcomes.
2. **Implementation & Refactoring**:
   - Write clean, modular, typed code adhering strictly to the architectural contracts.
3. **Execution of Automated Verification**:
   - Execute relevant `pytest` commands and capture terminal exit codes.
4. **Post-Phase Status Report**:
   - Document changed files, commands executed, test results, and remaining limitations.
