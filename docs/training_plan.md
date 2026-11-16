# AHIA-R2 Book Intelligence: Training Plan & Methodology

## 1. Core Principles & Constraints

In strict compliance with **Rule 1** (No Pretrained Language-Model Weights) and **Rule 5** (Pure PyTorch Implementation), the AHIA-R2 neural engine is initialized entirely with random Gaussian/Xavier weights and trained locally without external model checkpoints.

### Key Targets:
- **Architecture**: 8-block decoder-only Transformer ($d_{\text{model}}=256, n_{\text{heads}}=4, d_{\text{ff}}=1024, L_{\text{ctx}}=512, V=8000$).
- **Compute Target**: Local CPU training (AMD Ryzen 5 5500U, 12 vCPUs, multi-threaded PyTorch).
- **Phased Approach**: Dense baseline first; 4-expert MoE architecture activated only after dense baseline convergence.
- **Adaptive Depths**: Joint auxiliary loss heads at block 2, block 4, and block 8.

---

## 2. Two-Stage Training Curriculum

```
[Random Initialization (He / Xavier Normal)]
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ Stage 1: Causal Language Modeling (Pretraining) │
│ - Target: Domain fluency & financial syntax     │
│ - Corpus: Normalized book chunks & finance texts│
│ - Objective: Multi-depth causal next-token CE   │
└─────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ Stage 2: Grounded Synthesis Instruction Tuning  │
│ - Target: Context extraction & zero-fabrication │
│ - Corpus: Synthetic RAG context-query-response  │
│ - Objective: Grounded synthesis & refusal       │
└─────────────────────────────────────────────────┘
                 │
                 ▼
[Grounded Research Checkpoint (Dense Baseline)]
                 │
                 ▼ (Future Expansion)
[Stage 3: 4-Expert MoE Router Fine-Tuning]
```

---

## 3. Stage 1: Domain-Specific Causal Language Modeling

### 3.1 Loss Formulation with Multi-Depth Auxiliary Exits
To train adaptive execution depths at 2, 4, and 8 blocks, the network is trained with intermediate linear projections and auxiliary cross-entropy losses:

$$\mathcal{L}_{\text{total}} = \alpha_2 \mathcal{L}_2 + \alpha_4 \mathcal{L}_4 + \alpha_8 \mathcal{L}_8$$

Where:
- $\alpha_2 = 0.2, \alpha_4 = 0.3, \alpha_8 = 0.5$
- $\mathcal{L}_d = -\frac{1}{T} \sum_{t=1}^T \log P(x_t \mid x_{<t}; \theta_{1:d})$
- $\theta_{1:d}$ represents the parameters of the first $d$ Transformer blocks.

### 3.2 Optimization Hyperparameters
| Hyperparameter | Value | Description |
|---|---|---|
| Optimizer | AdamW | Pure PyTorch `torch.optim.AdamW` |
| Peak Learning Rate ($\eta_{\max}$) | $4.0 \times 10^{-4}$ | Standard for small Transformer from scratch |
| Minimum Learning Rate ($\eta_{\min}$) | $4.0 \times 10^{-5}$ | Cosine decay floor |
| Warmup Steps | 500 steps | Linear warmup from 0 to $\eta_{\max}$ |
| Weight Decay | 0.1 | Decoupled weight decay on non-bias parameters |
| Gradient Clipping | 1.0 | Global max gradient norm |
| Batch Size | 16–32 | Context window 512 tokens |
| Epochs | 10–20 epochs | Domain corpus coverage |

---

## 4. Stage 2: Grounded Synthesis & Refusal Fine-Tuning

### 4.1 Instruction & Grounding Template
The small model is fine-tuned to function as a strict context-conditioned extractor. Special tokens delineate the roles:

```
<query> What does Babylon recommend regarding savings? <context> A part of all you earn is yours to keep. It should be not less than a tenth no matter how little you earn. [CHUNK:BAB_CH02_0005] <response> Babylon recommends saving at least one-tenth of all you earn <cite> BAB_CH02_0005 </cite> </response>
```

### 4.2 Negative & Refusal Training Examples
To teach the model to never hallucinate or invent answers when context is lacking (Rule 10):

```
<query> What is Warren Buffett's opinion on quantum computing? <context> <nocontext> </context> <response> The provided library does not contain information to answer this question. </response>
```

Loss masking is applied: cross-entropy loss is computed **only on tokens in the `<response>` sequence**, leaving `<query>` and `<context>` unpenalized.

---

## 5. MoE Roadmap (Mixture-of-Experts)

After the dense baseline achieves stable convergence and passes all unit tests:
1. Replace dense feed-forward blocks at layers 4 through 8 with Top-1/Top-2 4-expert layers.
2. Initialize experts from the dense feed-forward weights with slight Gaussian perturbations.
3. Train gating routing networks using auxiliary load-balancing loss:
   $$\mathcal{L}_{\text{MoE}} = \mathcal{L}_{\text{task}} + \lambda_{\text{balance}} \sum_{i=1}^4 f_i \cdot P_i$$
   where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is the average routing probability.

---

## 6. CPU Compute Budget & Performance Envelope

- **Hardware**: AMD Ryzen 5 5500U (6 physical cores, 12 threads).
- **Forward-Backward Latency**: Estimated ~30–45 ms per batch step (batch size 16, sequence 512).
- **Total Training Time**: Estimated ~2.5 to 4 hours on local CPU for Stage 1 & Stage 2 on the 6-book finance corpus.
- **Inference Speed**: ~25–40 tokens/sec on CPU using stateful KV-caching.
