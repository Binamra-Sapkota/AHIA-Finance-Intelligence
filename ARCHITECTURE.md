# AHIA-R2 Book Intelligence: System Architecture

The complete, authoritative system architecture specification for AHIA-R2 Book Intelligence is located in:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

### Key Architectural Highlights
1. **Model**: Small Decoder-Only Transformer (~15.2M parameters) implemented in pure PyTorch and trained from random initialization.
2. **Retrieval**: Pure-Python Okapi BM25 indexing over a local library of six personal-finance books.
3. **Tokenizer**: Byte-level BPE tokenizer built from scratch with zero external model dependencies.
4. **Desktop GUI**: PySide6 desktop interface with thread-safe asynchronous dispatching.
5. **Reasoning Loop**: Deterministically bounded adaptive reasoning ($T_{max} \le 3$) with an independent outer wall-clock/memory supervisor.
6. **Grounding & Safety**: Strict citation ledger verifying zero fabrication. 100% offline runtime with no external AI API calls.

Please refer to [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full architectural blueprints, mathematical specifications, component diagrams, and the phased implementation roadmap.
