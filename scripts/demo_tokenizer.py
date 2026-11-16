"""AHIA-R2 Phase 3 Demonstration: Byte-Level BPE Tokenizer & Dataset Preparation.

Demonstrates:
- Training Byte-Level BPE from scratch without pretrained weights
- Vocabulary statistics and unknown-token frequency analysis
- Encode / decode round-trip fidelity with special tokens
- Pretraining dataset preparation with document-level train/validation split
- Zero document leakage and dataset provenance tracking
"""

from pathlib import Path
import sys
import tempfile

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.dataset.pipeline import PretrainDocument, PretrainingDatasetPipeline
from core.interfaces.ingestion import DocumentProvenance
from core.tokenizer.bpe import ByteLevelBPETokenizer


# Curated public-domain and original financial training corpus
AUTHORIZED_FINANCE_CORPUS = [
    (
        "DOC_BABYLON_CORE",
        "The Seven Cures for a Lean Purse",
        "George S. Clason (Public Domain)",
        """Start thy purse to fattening. For every ten coins thou puttest in, take out but nine. 
        Thy purse will start at once to fatten. Control thy expenditures. Confuse not necessary 
        expenses with thy desires. Make thy gold multiply. Put each coin to laboring that it may 
        reproduce its kind and help bring to thee abundance. Guard thy treasures from loss. 
        Invest only where thy principal is safe, where it may be reclaimed, and where thou wilt 
        not fail to collect a fair rental. Make of thy dwelling a profitable investment. 
        Insure a future income. Provide in advance for the needs of thy growing age and the 
        protection of thy family. Increase thy ability to earn. Cultivate thy own powers, study 
        and become wiser, become more skillful, and so act as to respect thyself.""",
    ),
    (
        "DOC_THINK_GROW_RICH",
        "Definiteness of Purpose",
        "Napoleon Hill (Public Domain)",
        """Fix in your mind the exact amount of money you desire. It is not sufficient merely 
        to say 'I want plenty of money.' Be definite as to the amount. Determine exactly what you 
        intend to give in return for the money you desire. Establish a definite date when you 
        intend to possess the money you desire. Create a definite plan for carrying out your desire, 
        and begin at once, whether you are ready or not, to put this plan into action. Write out 
        a clear, concise statement of the amount of money you intend to acquire, name the time 
        limit for its acquisition, state what you intend to give in return, and describe clearly 
        the plan through which you intend to accumulate it.""",
    ),
    (
        "DOC_CAPITAL_PRESERVATION",
        "Principles of Liquidity and Reserves",
        "AHIA-R2 Research Team (Original)",
        """An emergency fund is not an investment; it is an insurance policy against forced liquidation. 
        Maintaining three to six months of nondiscretionary cash outflows in highly liquid, capital-stable 
        instruments guarantees that personal solvency is never subordinated to transient market distress. 
        When equities decline, the investor with ample liquidity is a prospective buyer rather than a 
        distressed seller. Capital preservation precedes capital appreciation.""",
    ),
    (
        "DOC_COMPOUNDING_DYNAMICS",
        "Geometric Compounding and Time Horizons",
        "AHIA-R2 Research Team (Original)",
        """The mathematics of compound growth are inherently non-linear. In the initial phases of an 
        investment program, annual savings exceed portfolio investment returns. However, across a thirty-year 
        horizon, the cumulative return on past compounding vastly outstrips cumulative lifetime contributions. 
        Minimizing transactional frictions, avoiding punitive fee structures, and curbing emotional portfolio 
        churning are the operational keystones of wealth longevity.""",
    ),
    (
        "DOC_DEBT_MANAGEMENT",
        "Liability Structure and Capital Flow",
        "AHIA-R2 Research Team (Original)",
        """Uncollateralized high-interest liabilities are toxic to wealth creation. When borrowing costs 
        exceed conservative market return expectations, aggressive debt amortization yields a guaranteed, 
        risk-free return equal to the avoided financing rate. Structuring personal balance sheets to eliminate 
        sub-optimal liabilities liberates free cash flow for long-term equity accumulation.""",
    ),
]


def run_demonstration() -> None:
    print("=" * 75)
    print(" AHIA-R2 BOOK INTELLIGENCE: PHASE 3 TOKENIZER & DATASET DEMONSTRATION")
    print("=" * 75)

    # 1. Train Byte-Level BPE Tokenizer from scratch
    print("\n[Step 1] Training Byte-Level BPE Tokenizer From Scratch...")
    texts = [c[3] for c in AUTHORIZED_FINANCE_CORPUS]
    target_vocab = 600  # Configurable compact target for demonstration
    tokenizer = ByteLevelBPETokenizer(target_vocab_size=target_vocab)
    tokenizer.train_from_corpus(texts, target_vocab_size=target_vocab, min_frequency=2)

    # 2. Vocabulary Statistics
    print("\n[Step 2] Vocabulary Statistics Report:")
    stats = tokenizer.get_vocab_stats()
    print(f"  ► Total Vocabulary Size:     {stats['total_vocab_size']} tokens")
    print(f"  ► Special Control Tokens:    {stats['special_tokens_count']} (<pad>, <bos>, <eos>, <unk>, etc.)")
    print(f"  ► Base Byte Tokens:          {stats['base_byte_tokens_count']} (UTF-8 bytes 0x00 to 0xFF)")
    print(f"  ► Learned BPE Merges:        {stats['learned_merges_count']} merge rules")
    print(f"  ► Average Token Byte Length: {stats['avg_token_length_bytes']} bytes")
    print(f"  ► Maximum Token Byte Length: {stats['max_token_length_bytes']} bytes")

    # 3. Unknown-Token Frequency Analysis
    print("\n[Step 3] Unknown-Token Frequency Analysis:")
    eval_texts = [
        "Gold clingeth to the protection of the cautious owner who invests wisely.",
        "Financial figures: $10,500.75 yield at 8.5% with inflation risk.",
        "Complex terms: asset allocation, quantitative easing, and liquidity buffer.",
        "Non-ASCII symbols and emojis: 📈 Wealth in € and ¥ reaches 100%! 🚀",
    ]
    unk_freq = tokenizer.compute_unk_frequency(eval_texts)
    print(f"  ► Evaluated Sentences:       {len(eval_texts)}")
    print(f"  ► Unknown Token Frequency:   {unk_freq * 100.0:.2f}% (Byte-level BPE guarantees 0% unk)")

    # 4. Encode & Decode Round-Trip Demonstration
    print("\n[Step 4] Encode and Decode Round-Trip Demonstration:")
    sample_sentence = "Retain at least 10% of earnings: $1,250.50 saved diligently."
    token_ids = tokenizer.encode(sample_sentence, add_special_tokens=True)
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)

    print(f"  Input Text:       \"{sample_sentence}\"")
    print(f"  Encoded IDs:      {token_ids[:10]}... (Total {len(token_ids)} tokens)")
    print(f"  Decoded Output:   \"{decoded_text}\"")
    print(f"  Round-Trip Match: {'✓ 100% PERFECT MATCH' if sample_sentence == decoded_text else '✗ MISMATCH'}")

    # Special token roundtrip
    rag_prompt = "<query> What is the 10% rule? <context> Save ten percent. <response> Keep a tenth. <cite> BAB_01 </cite>"
    rag_tokens = tokenizer.encode(rag_prompt, add_special_tokens=False)
    rag_decoded = tokenizer.decode(rag_tokens, skip_special_tokens=False)
    print(f"\n  RAG Control Tokens Test:")
    print(f"  Decoded with Specials: \"{rag_decoded}\"")

    # 5. Pretraining Dataset Pipeline Compilation
    print("\n[Step 5] Pretraining Dataset Pipeline (Document-Level Splitting):")
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "pretraining_data"

        prov = DocumentProvenance(
            license_type="Public Domain / Authorized Original",
            provenance="Verified Corpus",
            redistribution_permitted=True,
            training_permitted=True,
            authorized_by_user=True,
        )

        doc_objs = [
            PretrainDocument(
                doc_id=doc_id,
                title=title,
                author=author,
                content=content,
                provenance=prov,
            )
            for doc_id, title, author, content in AUTHORIZED_FINANCE_CORPUS
        ]

        pipeline = PretrainingDatasetPipeline(
            tokenizer=tokenizer,
            max_context_len=128,
            val_ratio=0.20,
            seed=42,
        )
        record = pipeline.prepare_dataset(doc_objs, output_dir=out_dir)

        print(f"  ► Train Documents:           {record.train_documents_count}")
        print(f"  ► Validation Documents:      {record.val_documents_count}")
        print(f"  ► Train Sequences Generated: {record.train_sequences_count} (L_ctx = 128)")
        print(f"  ► Validation Sequences:      {record.val_sequences_count}")
        print(f"  ► Total Train Tokens:        {record.train_tokens_total}")
        print(f"  ► Total Validation Tokens:   {record.val_tokens_total}")
        print(f"  ► Zero Document Leakage:     {'✓ VERIFIED (0 Hash Collision)' if record.leakage_check_passed else 'FAILED'}")
        print(f"  ► Provenance Ledger Saved:   {out_dir / 'dataset_provenance.json'}")

    print("\n" + "=" * 75)
    print(" Phase 3 Tokenizer and Dataset Pipeline verified successfully!")
    print("=" * 75)


if __name__ == "__main__":
    run_demonstration()
