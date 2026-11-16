# AHIA-R2 Book Intelligence: Data Policy & Copyright Safe Harbor

## 1. Statutory Safe Harbor & Copyright Safeguards

In strict adherence to **Rule 11** (Strict Copyright Protection) and **Rule 12** (Git Repository Hygiene):

1. **Zero Redistribution of Copyrighted Materials**:
   - The AHIA-R2 Book Intelligence source code repository does not, and will never, distribute copyrighted book texts, proprietary datasets, or unauthorized excerpts.
   - All code, scripts, and documentation provided herein are open-source utilities designed to operate upon user-owned or public-domain texts.
2. **Local-Only Storage**:
   - All book files provided by the user must reside strictly on the local machine in `data/raw/` or `data/books/`.
   - Processed text chunks, SQLite databases, and BM25 indices reside in `data/processed/` and `data/indices/`.
   - All data directories are permanently excluded from Git tracking via `.gitignore`.
3. **Synthetic Test Fixtures**:
   - Unit, integration, and regression test suites in `tests/` must use exclusively synthetic mock sentences, algorithmic test patterns, or verifiable public-domain excerpts.
   - No copyrighted passages may appear in test assertions or code comments.

---

## 2. Supported Local Library Layout

Users configure their personal library within the local directory structure:

```
AHIA-Finance-Intelligence/
├── data/                      # 100% GIT-IGNORED
│   ├── raw/                   # User-provided book documents (PDF, EPUB, TXT)
│   ├── processed/             # Cleaned intermediate texts
│   ├── ahia_knowledge.db      # Local SQLite database storing catalog & chunks
│   └── indices/               # Serialized BM25 term matrices
```

### Curated Default Works:
1. *The Richest Man in Babylon* (George S. Clason) – Public Domain
2. *Think and Grow Rich* (Napoleon Hill) – Public Domain
3. Four slots for user-supplied modern personal-finance classics (e.g. *The Psychology of Money*, *The Intelligent Investor*, *Rich Dad Poor Dad*, *Your Money or Your Life*) provided locally by the user.

---

## 3. Privacy & Zero-Telemetry Guarantee

In strict compliance with **Rule 2** (No External AI APIs) and **Rule 8** (100% Offline Usability):
- No analytics, telemetry, crash reporting, or phone-home requests exist in the application code.
- No network sockets are opened during document ingestion, tokenization, BM25 retrieval, neural generation, or UI operation.
- The user maintains complete sovereign ownership of their reading history, indexed books, and generated query responses.

---

## 4. Citation Ledger & Anti-Fabrication Mandate

In strict compliance with **Rule 10** (Strict Grounding & Zero Fabrication):
- Any passage cited by the application must correspond byte-for-byte to an indexed chunk record in the local SQLite database.
- If a user queries a concept not present in their local indexed books, the system must explicitly decline to answer rather than fabricate or hallucinate content.
