# AGENTS.md - Engineering Governance & Operational Protocol
## Project: AHIA-R2 Book Intelligence

---

### 1. Mission & Architecture Identity

**AHIA-R2 Book Intelligence** is an open-source research demonstration that builds a compact, self-contained desktop AI application powered by a small decoder-only Transformer trained entirely from random initialization (no pretrained weights, no external LLM APIs, no pretrained embeddings). The system indexes and grounds its knowledge upon a curated local library of six personal-finance books using classical Okapi BM25 retrieval combined with an adaptive, bounded reasoning controller and a PySide6 desktop interface.

All contributing human engineers and AI agents must strictly adhere to the policies, boundaries, and architectural guidelines defined in this document.

---

### 2. The 15 Non-Negotiable Operational Rules

Every code change, design decision, pull request, and operational action is governed by these fifteen rules:

1. **No Pretrained Language-Model Weights**: All neural model weights must be trained from random initialization. Pretrained weights (e.g. HuggingFace GPT-2, LLaMA, BERT, Mistral) are strictly forbidden.
2. **No External AI APIs in Runtime**: The finished desktop application must run fully autonomous on the user's local machine without calling OpenAI, Anthropic, Google, Cohere, or any external AI services.
3. **No Pretrained Embedding Models**: Semantic embeddings cannot come from pretrained models (e.g., sentence-transformers, OpenAI embeddings, fastText). Initial retrieval is driven by BM25, and representation learning is self-contained.
4. **Custom Tokenizer Built from Scratch**: The tokenizer must be trained from scratch on an authorized text corpus using an explicit, inspectable algorithm (e.g., Byte-level BPE).
5. **Pure PyTorch Implementation**: All neural network layers, attention mechanisms, loss functions, optimizers, and training loops must be implemented using PyTorch (`torch`, `torch.nn`).
6. **BM25 for Initial Retrieval**: Initial document selection and candidate generation from the book corpus must use Okapi BM25 (or an authorized variant such as BM25+ / BM25L).
7. **PySide6 for Desktop Application**: The graphical user interface must be developed using PySide6 (Qt for Python).
8. **100% Offline Usability**: After environment setup and local training/indexing, the application must function with zero network access. No telemetry, no remote asset fetching.
9. **Zero Exaggeration of Intelligence**: Never claim commercial-model-level intelligence (e.g. "GPT-4 equivalent"). Explicitly communicate the model's small footprint, bounded domain scope, and architectural nature to users.
10. **Strict Factual Grounding & Zero Fabrication**: Never fabricate quotations, book passages, page numbers, chapter titles, or citations. If a requested concept is missing from the retrieved context, the system must state that the knowledge library does not contain the answer.
11. **Strict Copyright Protection**: Never redistribute copyrighted book texts or unauthorized datasets. Raw books and copyrighted materials must never be committed to source control.
12. **Git Repository Hygiene**: Private book files (`data/raw/`, `data/books/`), processed training corpora (`data/processed/`), indices (`data/indices/`), and model checkpoints (`checkpoints/`, `models/*.pt`) must be kept out of public Git commits via `.gitignore`.
13. **Full Test & Training Coverage for Adaptive Paths**: Every adaptive execution path (e.g., query reformulation, multi-hop context retrieval, fallback branches) must have explicit unit tests and appropriate training/verification coverage.
14. **Deterministic Reasoning Loop Limits**: All iterative retrieval and reasoning loops must have a strictly enforced deterministic maximum iteration limit ($T_{max}$, e.g., 3 iterations). Infinite loops or unbounded search paths are prohibited.
15. **Independent Resource Limit Enforcement**: System resource limits (CPU timeout ceiling, memory bounds, token generation limits) must be enforced by an independent outer supervisor, not left solely to the learned neural controller or model stop tokens.

---

### 3. Engineering & Code Quality Standards

1. **Modular Python Architecture**:
   - Clean layer separation: `core/` (data ingestion, tokenization, neural modeling, retrieval, reasoning), `service/` (concurrency, lifecycle, DTOs), `ui/` (PySide6 components, views, models), and `tests/`.
   - The UI layer must never import PyTorch or tensor data structures directly. Data crosses the UI/core boundary via typed Dataclasses (Data Transfer Objects).
2. **Type Hints**:
   - 100% of public and internal functions, methods, and class attributes must have comprehensive Python type hints (`typing`, `Optional`, `List`, `Dict`, `Tuple`, `Generator`, `Protocol`, `Callable`).
3. **Documentation & Docstrings**:
   - All modules, public classes, and functions must have Google-style or Sphinx-style docstrings specifying Arguments, Returns, and Exceptions.
4. **Pytest Suite**:
   - Every module must have automated tests located in `tests/`.
   - Tests must run cleanly with `pytest`.
   - No mock replacements of failing implementations: do not paper over broken logic with stub mocks. Fix the root cause or document genuine architectural constraints.
   - Do not claim tests passed without executing them and recording the terminal output.
5. **Dependency Discipline**:
   - Minimize dependencies. Rely on PyTorch, PySide6, standard library, and lightweight algorithmic packages (e.g., `pytest`, `numpy`). Avoid heavy or bloated frameworks.
6. **Data Integrity & Non-Destruction**:
   - Never delete or overwrite user data or configuration files without explicit user consent.
   - Always confirm before deleting files or making major architectural pivots.

---

### 4. Agent Roles & Specialization Matrix

When subagents or collaborative workers are deployed on this project, they must assume designated roles:

| Role | Domain / Ownership | Primary Deliverables |
|---|---|---|
| **ML Architect & Lead** | System architecture, end-to-end integration, performance budgets | `ARCHITECTURE.md`, module interfaces, training strategy |
| **Data & Retrieval Specialist** | Book ingestion, chunking, metadata extraction, BM25 indexing | `core/data/`, `core/retrieval/`, citation ledger |
| **Tokenizer & Modeling Specialist** | Custom BPE tokenizer, decoder-only Transformer, KV-cache | `core/tokenizer/`, `core/model/`, `core/training/` |
| **Reasoning & Safety Controller** | Bounded reasoning loop, citation validator, resource limiter | `core/reasoning/`, `core/safety/`, supervisor guards |
| **Desktop Application Engineer** | PySide6 UI, thread isolation, async workers, user experience | `ui/`, `service/`, reactive state management |
| **Verification & Quality Engineer** | Pytest suites, regression tests, deterministic limit validation | `tests/`, benchmark scripts, test reporting |

---

### 5. Phased Development Lifecycle Protocol

To maintain complete transparency and rigorous quality, every development step must follow this 4-step protocol:

1. **Pre-Phase Implementation Plan**:
   - Before executing code changes, state the specific phase, goals, target files, and expected test outcomes.
2. **Implementation & Refactoring**:
   - Implement clean, modular, typed code adhering to the architectural specification.
3. **Execution of Automated Verification**:
   - Run relevant `pytest` suites or verification commands in the terminal and capture exact exit codes and output.
4. **Post-Phase Status Report**:
   - Document changed files, commands executed, test results, and any limitations or next actions.

---

### 6. Copyright & Privacy Safe Harbor

- All book texts used for local personal finance assistance must reside strictly on the user's local machine in `data/raw/` or `data/books/`.
- Neither raw books nor copyrighted passages shall be committed to the public Git repository.
- Synthesized examples in test suites must use synthetic mock sentences or public-domain excerpts, not copyrighted excerpts.
