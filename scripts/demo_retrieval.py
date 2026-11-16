"""AHIA-R2 Phase 2 Demonstration: Local Ingestion and BM25 Retrieval Preparation.

Demonstrates local parsing, chunking, SQLite storage, and BM25 ranking
using original sample personal-finance text safe for open-source redistribution.
"""

from pathlib import Path
import sys
import tempfile

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.ingestion.pipeline import IngestionPipeline
from core.ingestion.repository import BookRepository
from core.interfaces.ingestion import DocumentProvenance
from core.retrieval.bm25 import BM25Index


SAMPLE_REDISTRIBUTABLE_TEXT = """# Foundations of Financial Intelligence

The foundational axiom of personal wealth accumulation is the principle of the surplus: 
you must consistently spend less than you earn. Retaining a minimum of ten percent of all 
gross earnings creates the primary seed capital necessary for financial autonomy. Without 
this initial retained margin, no investment compounding can ever occur.

An emergency reserve represents the primary fortress of capital preservation. Before allocating 
capital to volatile risk assets, every individual should establish liquid reserves covering 
three to six months of basic living expenditures. This buffer guarantees that market downturns 
or unexpected interruptions in employment do not force the premature liquidation of long-term assets.

Compound interest is an exponential mechanism where reinvested earnings generate further earnings. 
Over multi-decade investment horizons, the geometric progression of reinvested dividends and capital 
gains dramatically outpaces the sum of lifetime principal contributions. Time and patience are 
the investor's most potent compounding multipliers.

High-interest consumer debt represents an aggressive drag on capital formation. Carrying revolving 
credit liabilities at exorbitant interest rates guarantees a negative real return on your labor. 
Eliminating high-cost obligations should always take precedence over speculative investments.

Asset allocation and broad diversification reflect epistemic humility. Because economic cycles 
and sector rotations cannot be predicted with consistency, distributing investment capital across 
uncorrelated asset classes shields portfolios from catastrophic single-point failure.
"""


def run_demonstration() -> None:
    print("=" * 75)
    print(" AHIA-R2 BOOK INTELLIGENCE: PHASE 2 LOCAL RETRIEVAL DEMONSTRATION")
    print("=" * 75)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sample_file = tmp_path / "foundations_of_financial_intelligence.md"
        sample_file.write_text(SAMPLE_REDISTRIBUTABLE_TEXT, encoding="utf-8")

        db_path = tmp_path / "demo_knowledge.db"
        repo = BookRepository(db_path=db_path)
        bm25 = BM25Index()
        pipeline = IngestionPipeline(repository=repo, bm25_index=bm25, chunk_target_words=60, chunk_overlap_words=15)

        provenance = DocumentProvenance(
            license_type="Open Source Research / Public Domain",
            provenance="Original Research Synthesis",
            redistribution_permitted=True,
            training_permitted=True,
            authorized_by_user=True,
        )

        print("\n[Step 1] Ingesting Document...")
        print(f"  Source File: {sample_file.name}")
        print(f"  Provenance:  {provenance.license_type} ({provenance.provenance})")
        print(f"  Permissions: Redistribution={provenance.redistribution_permitted}, Training={provenance.training_permitted}")

        book_meta, chunks = pipeline.ingest_file(
            file_path=sample_file,
            title="Foundations of Financial Intelligence",
            author="AHIA-R2 Research Team",
            provenance=provenance,
        )

        print(f"\n[Step 2] Ingestion Complete:")
        print(f"  Assigned Doc ID:    {book_meta.book_id}")
        print(f"  SHA-256 Digest:     {book_meta.file_hash[:16]}...")
        print(f"  Total Words:        {book_meta.total_words}")
        print(f"  Generated Chunks:   {len(chunks)}")
        print(f"  SQLite Persistence: Successfully saved to {db_path.name}")

        test_queries = [
            "What percentage of income should be retained for savings?",
            "Why is an emergency reserve necessary before investing?",
            "How does compound interest create long term wealth?",
            "What is the impact of high interest consumer debt?",
        ]

        print("\n" + "=" * 75)
        print(" [Step 3] Executing BM25 Information Retrieval Queries")
        print(" (Top passages selected strictly via Okapi BM25; no answer generation)")
        print("=" * 75)

        for q_idx, query in enumerate(test_queries, start=1):
            print(f"\nQuery #{q_idx}: \"{query}\"")
            results = bm25.search(query, top_k=1)

            if results:
                match = results[0]
                print(f"  ► Relevance Score:   {match.score:.3f}")
                print(f"  ► Reference Label:   {match.reference_label}")
                print(f"  ► Chunk ID:          {match.chunk.chunk_id}")
                print(f"  ► Retrieved Passage:")
                print(f"    \"{match.chunk.content}\"")
            else:
                print("  ► No relevant passage found above threshold.")

    print("\n" + "=" * 75)
    print(" Demonstration completed successfully with 100% offline local retrieval.")
    print("=" * 75)


if __name__ == "__main__":
    run_demonstration()
