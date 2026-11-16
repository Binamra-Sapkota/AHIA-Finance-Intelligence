# AHIA-R2 Book Intelligence: System Architecture Specification

## 1. Executive Summary & Project Objectives

**AHIA-R2 Book Intelligence** is an open-source, fully offline, desktop research demonstration that couples classical Information Retrieval (Okapi BM25) with a custom, small decoder-only Transformer trained entirely from random initialization (zero pretrained weights, zero pretrained embeddings, zero external APIs). The system serves as a grounded research and question-answering assistant over a curated local library of six personal-finance books.

### Key Operational Constants
- **Weight Provenance**: 100% trained from random initialization. Pretrained model checkpoints (e.g., GPT-2, LLaMA, BERT) are strictly forbidden.
- **Runtime Connectivity**: 100% offline. Zero external API calls, zero telemetry, zero web fetching.
- **Hardware Target**: Consumer x86_64 CPU (AMD Ryzen 5 5500U, 6 cores / 12 threads, 8 GB RAM) running Linux/Windows. No GPU required.
- **Grounding Integrity**: Zero fabrication of citations, page numbers, or book passages. If retrieved context does not support an answer, the system gracefully declines.
- **Bound on Complexity**: Deterministic reasoning loops ($T_{max} \le 3$), resource limits enforced by an external supervisor.
- **Copyright Safe Harbor**: Raw book texts and training checkpoints are strictly excluded from version control.

---

## 2. System Architecture & Component Decomposition

The system is organized into modular tiers ensuring total separation of concerns, complete testability, and decoupled UI:

```
AHIA-Finance-Intelligence/
├── AGENTS.md                  # Operational and governance rules for AI agents and engineers
├── ARCHITECTURE.md            # Architecture specification pointer
├── .gitignore                 # Exclusion of copyrighted books, checkpoints, and caches
├── pyproject.toml / requirements.txt # Minimal dependency definitions
├── docs/
│   └── ARCHITECTURE.md        # Comprehensive technical architecture specification
├── data/                      # Local data directory (git-ignored)
│   ├── raw/                   # Raw user-provided book files (PDF, EPUB, TXT)
│   ├── processed/             # Cleaned, segmented chunks with citation metadata
│   └── indices/               # Serialized BM25 inverted index and vocabularies
├── checkpoints/               # Local PyTorch weights and optimizer states (git-ignored)
├── core/                      # Pure Machine Learning & Algorithmic Core (NO UI DEPENDENCIES)
│   ├── __init__.py
│   ├── data/                  # Ingestion, parsing, normalization, chunking
│   │   ├── __init__.py
│   │   ├── parser.py          # Multi-format document parser (PDF, EPUB, TXT)
│   │   ├── chunker.py         # Metadata-preserving sliding-window chunker
│   │   └── schema.py          # Chunk, BookMetadata, CitationRef data schemas
│   ├── tokenizer/             # Custom Byte-Level BPE Tokenizer from Scratch
│   │   ├── __init__.py
│   │   ├── bpe.py             # Pure-Python BPE trainer and encoder/decoder
│   │   ├── vocab.py           # Vocabulary, special tokens, serialization
│   │   └── special_tokens.py  # <pad>, <bos>, <eos>, <query>, <context>, <response>, <cite>
│   ├── retrieval/             # Information Retrieval Subsystem
│   │   ├── __init__.py
│   │   ├── bm25.py            # Okapi BM25 / BM25+ indexer and search engine
│   │   ├── indexer.py         # Corpus indexing pipeline and inverted index storage
│   │   └── ranker.py          # Score normalization, thresholding, candidate ranking
│   ├── model/                 # Pure PyTorch Decoder-Only Transformer
│   │   ├── __init__.py
│   │   ├── config.py          # TransformerConfig dataclass (hyperparameters)
│   │   ├── attention.py       # Causal Multi-Head Self-Attention + KV-Cache
│   │   ├── layers.py          # RMSNorm, SwiGLU / GELU Feed-Forward, Rotary/Learned Positional
│   │   ├── transformer.py     # Full Decoder-Only Model (PyTorch nn.Module)
│   │   └── generation.py      # Autoregressive generation engine with KV-cache
│   ├── training/              # Training & Fine-Tuning Pipeline
│   │   ├── __init__.py
│   │   ├── dataset.py         # PyTorch Dataset for causal LM and context-synthesis
│   │   ├── trainer.py         # Pure PyTorch Trainer (CPU optimized, gradient clipping)
│   │   └── synthetic.py       # Synthesis dataset generator for context-conditioned QA
│   ├── reasoning/             # Adaptive Bounded Reasoning Controller
│   │   ├── __init__.py
│   │   ├── controller.py      # Deterministic bounded loop manager (T_max = 3)
│   │   ├── query_analyzer.py  # Query decomposition, intent detection, reformulation
│   │   └── citation_ledger.py # Grounding verification and quote validator
│   └── safety/                # Independent Safety & Resource Supervision
│       ├── __init__.py
│       ├── supervisor.py      # External wall-clock timeout & memory monitor
│       └── boundaries.py      # Capability disclaimers and refusal handlers
├── service/                   # Application Service Layer (Decoupling Core & UI)
│   ├── __init__.py
│   ├── dto.py                 # Typed Data Transfer Objects (QueryRequest, QueryResult, etc.)
│   ├── worker.py              # Qt QRunnable / QThread async dispatchers
│   └── app_service.py         # Facade coordinating core components for UI consumption
├── ui/                        # PySide6 Desktop Application (NO PYTORCH IMPORTS)
│   ├── __init__.py
│   ├── main_window.py         # Primary PySide6 application window
│   ├── components/            # Reusable UI widgets
│   │   ├── chat_view.py       # Conversation canvas with streaming response
│   │   ├── citation_panel.py  # Source book inspector & excerpt viewer
│   │   ├── library_manager.py # Book indexing status and chunk explorer
│   │   └── metrics_widget.py  # CPU usage, latency, tokens/s, iteration count
│   ├── styles/                # Modern desktop theme and CSS
│   └── assets/                # Local UI icons and styling assets
└── tests/                     # Comprehensive Automated Pytest Suite
    ├── __init__.py
    ├── conftest.py            # Shared fixtures (synthetic test documents, temp dirs)
    ├── test_data.py           # Ingestion, parsing, and chunking tests
    ├── test_tokenizer.py      # BPE training, encoding, decoding, roundtrip tests
    ├── test_retrieval.py      # BM25 indexing, ranking, and citation ledger tests
    ├── test_model.py          # Transformer forward pass, KV cache, and shapes
    ├── test_training.py       # Loss convergence, checkpoint save/load
    ├── test_reasoning.py      # Bounded loop limits, reformulation, fallback paths
    ├── test_safety.py         # Timeout tripwire, memory ceiling, refusal guarantees
    └── test_dto_contracts.py  # DTO contract and service decoupling tests
```

---

## 3. Data Ingestion & Library Management

### 3.1 The Six-Book Local Finance Library
The application is tailored for personal-finance literature. The default library supports six foundational works:
1. *The Richest Man in Babylon* (George S. Clason) – Public Domain
2. *Think and Grow Rich* (Napoleon Hill) – Public Domain
3. Four user-provided personal finance classics (e.g. *The Psychology of Money*, *The Intelligent Investor*, *Rich Dad Poor Dad*, *Your Money or Your Life*) loaded locally.

### 3.2 Ingestion Engine (`core/data/`)
- **Parsers**: Native text parsing for `.txt`, lightweight extraction for `.pdf` (using pure-Python PDF text extraction) and `.epub` (HTML/XML parsing).
- **Text Normalization**: Unicode NFKC normalization, whitespace consolidation, hyphenation rejoining across line wraps, preservation of chapter headers.
- **Sliding-Window Chunking**:
  - Target window: 250 words (~320 tokens).
  - Overlap: 50 words (~65 tokens).
  - Boundary sensitivity: Chunks break at sentence or paragraph boundaries when within 10% of window limit.
- **Metadata Preservation (`schema.py`)**:
  ```python
  @dataclass(frozen=True)
  class Chunk:
      chunk_id: str             # e.g., "RICHEST_BABYLON_CH03_0014"
      book_id: str              # e.g., "RICHEST_BABYLON"
      book_title: str           # e.g., "The Richest Man in Babylon"
      author: str               # e.g., "George S. Clason"
      chapter_title: str        # e.g., "Seven Cures for a Lean Purse"
      section_title: str        # Optional sub-heading
      char_start: int           # Character offset in source
      char_end: int             # Character offset end in source
      content: str              # Normalized chunk text
      token_count: int          # Token length
  ```

---

## 4. Custom Byte-Level BPE Tokenizer (`core/tokenizer/`)

In adherence to Rule 4, no pretrained tokenizer or vocabulary is used. A Byte-level BPE tokenizer is implemented and trained from scratch on authorized text:

### 4.1 Specifications
- **Base Alphabet**: 256 individual byte tokens (values 0–255), guaranteeing that any valid UTF-8 byte stream can be encoded with zero `<unk>` fallback crashes.
- **Target Vocabulary Size**: 4,096 to 8,192 tokens (optimized for small parameter models and fast CPU inference).
- **Special Control Tokens**:
  - `<pad>` (ID: 0) – Padding for batching.
  - `<bos>` (ID: 1) – Beginning of sequence.
  - `<eos>` (ID: 2) – End of sequence.
  - `<unk>` (ID: 3) – Out of vocabulary fallback.
  - `<query>` (ID: 4) – Query prefix delimiter.
  - `<context>` (ID: 5) – Retrieved context block delimiter.
  - `<response>` (ID: 6) – Model response prefix.
  - `<cite>` (ID: 7) – Citation annotation marker.
  - `<nocontext>` (ID: 8) – Fallback marker when retrieval finds no grounding.
- **Serialization**: Standalone JSON vocabulary dictionary and merge-pair table. Fully inspectable, deterministic, and self-contained.

---

## 5. Information Retrieval Subsystem (`core/retrieval/`)

In adherence to Rule 6, initial book retrieval uses Okapi BM25.

### 5.1 BM25 Mathematical Formulation
For query $Q = \{q_1, q_2, \dots, q_n\}$ and document chunk $D$:
$$\text{Score}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
Where:
- $\text{IDF}(q_i) = \ln\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)$
- $k_1 = 1.5$ (term frequency saturation parameter)
- $b = 0.75$ (document length penalization parameter)
- $|D|$ is the length of chunk $D$ in tokens, and $\text{avgdl}$ is the average chunk length across the corpus.

### 5.2 Retrieval Pipeline & Grounding Ledger
1. **Query Tokenization & Normalization**: Strips punctuation, applies lowercasing and domain-specific stopword filtering.
2. **Inverted Index Lookup**: Fast candidate accumulator scoring all matching chunks.
3. **Top-$K$ Selection**: Extracts top $K \in [3, 5]$ chunks based on BM25 score.
4. **Relevance Thresholding**:
   - If $\max(\text{Score}) < \theta_{min}$ (e.g., threshold score = 2.0), the engine flags retrieval as *insufficient*.
   - The citation ledger enforces that every context provided to the generator has a verified `Chunk` source.

---

## 6. Small Decoder-Only Transformer Architecture (`core/model/`)

In adherence to Rule 1 and Rule 5, the neural model is a decoder-only Transformer written in pure PyTorch, trained from random initialization.

### 6.1 Architectural Parameters (CPU-Optimized)
| Hyperparameter | Value | Rationale |
|---|---|---|
| Vocabulary Size ($V$) | 4,096 | Compact embedding table ($4096 \times 288 \approx 1.18\text{M}$ params) |
| Model Dimension ($d_{model}$) | 288 | Balances representational power and CPU latency |
| Number of Layers ($n_{layers}$) | 6 | Shallow depth keeps forward pass under 25ms on CPU |
| Attention Heads ($n_{heads}$) | 6 ($d_k = 48$) | Multi-head factorization matching $d_{model}$ |
| Feed-Forward Dim ($d_{ff}$) | 1,152 ($4 \times d_{model}$) | Standard capacity multiplier |
| Context Window ($L_{ctx}$) | 512 tokens | Sufficient for 1-2 chunks of context + query + answer |
| Activation | SwiGLU / GELU | Smooth gradient propagation |
| Normalization | RMSNorm (Pre-Norm) | Numerical stability without mean calculation overhead |
| Positional Encoding | Rotary Embeddings (RoPE) | Relative position awareness and length extrapolation |
| Total Parameter Count | **~15.2 Million** | Footprint: ~60 MB FP32, ~30 MB FP16. Fast on Ryzen 5 CPU |

### 6.2 Key Subcomponents
1. **Rotary Position Embedding (RoPE)**: Applied to query and key vectors before dot-product attention, allowing natural relative distance modeling without learned absolute positional tables.
2. **Causal Masked Self-Attention with KV-Cache**:
   - During training: Lower-triangular causal attention mask.
   - During inference: Step-by-step decoding with cached Key and Value states across past tokens, reducing step complexity from $O(n^2)$ to $O(n)$.
3. **RMSNorm**:
   $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$

---

## 7. Training & Curriculum Strategy (`core/training/`)

Because the model starts from random initialization and has ~15M parameters, training must be strictly targeted and efficient:

### 7.1 Stage 1: Domain-Specific Causal Language Modeling
- **Corpus**: Tokenized chunks of the six finance books and related public-domain finance texts.
- **Objective**: Standard next-token prediction (Cross-Entropy loss):
  $$\mathcal{L}_{CLM} = -\sum_{t=1}^T \log P(x_t \mid x_{<t}; \theta)$$
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.95, \text{weight\_decay}=0.1$).
- **Learning Rate Schedule**: Cosine decay with 500-step linear warmup, peak learning rate $3 \times 10^{-4}$, minimum $3 \times 10^{-5}$.
- **Gradient Clipping**: Max norm 1.0.

### 7.2 Stage 2: Context-Conditioned Synthesis & Grounding Fine-Tuning
- **Objective**: Teach the small model to behave as an extractive synthesizer rather than an ungrounded generator.
- **Prompt Structure**:
  ```
  <query> {USER_QUERY} <context> {RETRIEVED_CHUNKS} <response> {GROUNDED_SYNTHESIS} <cite> {CHUNK_ID} </cite>
  ```
- **Negative / Fallback Training**:
  ```
  <query> {OUT_OF_DOMAIN_QUERY} <context> <nocontext> </context> <response> The provided library does not contain information to answer this question. </response>
  ```
- This ensures the model learns to output refusal when context is absent or irrelevant.

---

## 8. Bounded Reasoning Controller & Safety Architecture (`core/reasoning/`, `core/safety/`)

In adherence to Rule 13, 14, and 15:

### 8.1 The Bounded Reasoning Loop ($T_{max} = 3$)
The reasoning controller executes an adaptive loop with a deterministic upper bound:

```
[User Query] 
     │
     ▼
[Step 1: Query Analysis] ──► Extract financial terms & intent
     │
     ▼
[Step 2: BM25 Retrieval] ──► Retrieve Top-K candidates
     │
     ├─► Are candidates confident? (Score >= Threshold)
     │       ├─► YES ──► Proceed to Step 4 (Neural Generation)
     │       └─► NO  ──► Iteration < T_max?
     │                     ├─► YES: [Step 3: Query Reformulation] ──► Loop to Step 2
     │                     └─► NO:  Graceful Fallback ("No relevant books found")
     ▼
[Step 4: Context Conditioning] ──► Feed [Query + Chunks] to Transformer
     │
     ▼
[Step 5: Citation & Grounding Check]
     │
     ├─► Are cited claims supported by chunk text?
     │       ├─► Valid: Return Grounded Response + Citations
     │       └─► Invalid / Hallucination detected: Prune or replace with verbatim excerpt
     ▼
[Result to Service Layer]
```

### 8.2 Deterministic Termination Guarantees
1. **Loop Bound**: Under no circumstances can the reasoning loop exceed $T_{max} = 3$ iterations.
2. **Independent Wall-Clock Supervisor (`core/safety/supervisor.py`)**:
   - A dedicated Python timer thread/watchdog enforces a hard timeout ceiling ($T_{timeout} = 10.0$ seconds). If the model or retrieval stalls, the supervisor terminates execution and returns a timeout fallback.
3. **Generation Ceiling**: Maximum generation tokens strictly capped at $N_{max} = 256$ tokens.
4. **Anti-Fabrication Guard**: The `citation_ledger.py` scans model outputs for any generated `<cite>` tag or quote. If the quoted string does not have a high fuzzy match ($\ge 85\%$) with the retrieved chunk text, it is flagged and suppressed.

---

## 9. Service Layer & Decoupling (`service/`)

To prevent UI locking and ensure complete decoupling:
- **DTOs (`service/dto.py`)**:
  - `QueryRequest`: User prompt, search preferences, optional filter by book.
  - `QueryResult`: Answer text, citation list (`CitationItem`), iteration count, latency ms, confidence score.
  - `CitationItem`: Book title, author, chapter, section, chunk excerpt, confidence.
  - `SystemStatus`: Model loaded flag, index size, memory footprint.
- **Worker Concurrency (`service/worker.py`)**:
  - PySide6 `QRunnable` executed on a global `QThreadPool`.
  - Emits Qt signals (`started`, `token_stream`, `citations_ready`, `completed`, `error`) to update UI smoothly on the main GUI thread.

---

## 10. Desktop User Interface Architecture (`ui/`)

In adherence to Rule 7 and Rule 8, the GUI is built with PySide6:

### 10.1 UI Views
1. **Query & Exploration Canvas**:
   - Clean, modern conversation thread.
   - Streaming text rendering as tokens are generated.
   - Clickable citation chips (e.g. `[Babylon, Ch. 3]`) linking directly to source passages.
2. **Citation & Source Inspector (Right Drawer)**:
   - Displays full source chunk text with matching keywords highlighted.
   - Shows metadata: Book Title, Chapter, Paragraph Offset, and BM25 Score.
3. **Knowledge Library Manager**:
   - Visual inventory of the six personal finance books.
   - Status indicators (Indexed, Word Count, Chunk Count).
   - "Rebuild BM25 Index" button with progress bar.
4. **System Diagnostics Bar**:
   - Live metrics: Memory consumption (MB), CPU inference latency (ms), Generation speed (tokens/sec), Active reasoning iterations.
   - Transparent disclaimer badge: *"AHIA-R2 15M Small Research Model • 100% Offline • Strictly Grounded"*.

---

## 11. Verification & Testing Strategy (`tests/`)

Every phase must pass automated pytest tests before completion:
- `test_data.py`: Multi-format parsing, chunking consistency, metadata completeness.
- `test_tokenizer.py`: BPE merge table generation, byte-level fallback, special token handling, encode/decode roundtrips.
- `test_retrieval.py`: BM25 scoring accuracy, top-$K$ sorting, thresholding, empty query handling.
- `test_model.py`: Transformer dimension shapes, causal masking, KV-cache output invariance (verifying that KV-cached decoding matches full causal forward pass).
- `test_training.py`: Training step loss descent, checkpoint save and restore verification.
- `test_reasoning.py`: Enforcing $T_{max} \le 3$, query reformulation path, fallback when no context found.
- `test_safety.py`: Independent supervisor timeout triggering, anti-fabrication citation verification.
- `test_dto_contracts.py`: Verifying zero PyTorch dependencies in DTOs and UI interfaces.

---

## 12. Implementation Roadmap & Phased Execution

- **Phase 1**: Project Environment Setup, Minimal Dependencies & Core Typing Schemas.
- **Phase 2**: Document Ingestion, Clean Text Normalization & Metadata Chunking.
- **Phase 3**: Custom Pure-Python Okapi BM25 Retrieval Engine & Inverted Indexer.
- **Phase 4**: Custom Byte-Level BPE Tokenizer Engine (From Scratch).
- **Phase 5**: Small Decoder-Only Transformer (PyTorch) with Causal Attention & KV-Cache.
- **Phase 6**: Training Pipeline & Synthetic Grounded Synthesis Fine-Tuning.
- **Phase 7**: Bounded Reasoning Controller, Citation Ledger & Independent Resource Supervisor.
- **Phase 8**: PySide6 Desktop GUI & Service Integration.
- **Phase 9**: End-to-End Verification, Pytest Test Suite & Final System Documentation.
