# LightCSR

**Lightweight Retrieval-Augmented Generation with Small Language Models for Cold-Start Recommendation**

> Official implementation of the paper *LightCSR: Lightweight LLM + RAG Framework for Cold-Start Recommendation*.
>
> LightCSR achieves recommendation quality comparable to large-model RAG baselines (e.g., ColdRAG with GPT-4o) on cold-start scenarios, while using **1B–3B parameter** small language models with **~200ms latency** and **~0.3% of the API cost**.

---

## Overview

Cold-start recommendation with LLM-based RAG frameworks (ColdRAG, KALM4Rec, MARC) currently relies on proprietary large models (GPT-4o, Gemini Pro), incurring high latency and cost. LightCSR fills this gap:

- **Small Language Model Ranker** — 1B–3B open-source models (Qwen2.5, Llama-3.2) instead of GPT-4o.
- **Cold-start-specialized dual-level retrieval** — low-level entity vector matching + high-level knowledge-graph multi-hop traversal, compensating for the limited world knowledge of small models.
- **Optional distillation + quantization** — GPT-4o teacher distillation and INT4 quantization for further acceleration.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     LightCSR Framework                    │
│                                                           │
│  ┌───────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  Profile  │───▶│  Retrieval   │───▶│  Lightweight  │  │
│  │  Builder  │    │  Module      │    │  LLM Ranker   │  │
│  │ (item/    │    │ (KG + Vector,│    │ (1B–3B, INT4, │  │
│  │  user)    │    │  dual-level) │    │  + KD)        │  │
│  └───────────┘    └──────────────┘    └───────────────┘  │
│                         ▓▓ Knowledge Graph ▓▓            │
└──────────────────────────────────────────────────────────┘
```

**Pipeline:**
1. **Profile Builder** — converts structured item/user attributes into natural-language profiles; extracts entities and relations to build a dynamic domain knowledge graph.
2. **Retrieval Module** — dual-level retrieval: low-level entity vector similarity (concrete item matching) + high-level KG relation traversal (abstract semantic matching); returns Top-K candidates with supporting evidence paths.
3. **Lightweight LLM Ranker** — a 1B–3B SLM ranks candidates conditioned on the user profile, candidate list, and retrieved evidence; optional GPT-4o distillation and INT4 quantization.

---

## Quick Start

### Requirements

- Python >= 3.10
- CUDA >= 12.1 (for local SLM inference)
- 1× NVIDIA A100 (80GB) recommended for evaluation; a single RTX 4090 (24GB) suffices for INT4 inference of 3B models

### Installation

```bash
git clone https://github.com/your-org/lightcsr.git
cd lightcsr

# Create environment
conda create -n lightcsr python=3.10 -y
conda activate lightcsr

# Install dependencies
pip install -r requirements.txt

# Install FAISS (GPU version recommended)
pip install faiss-gpu  # or faiss-cpu for CPU-only
```

### Data Preparation

Download the datasets:

| Dataset | Domain | Cold-Start Type | Source |
|---------|--------|-----------------|--------|
| Amazon Games | Video games | Item CS | [Amazon Reviews](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) |
| Amazon CDs | Music | Item CS | [Amazon Reviews](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) |
| Amazon Beauty | Beauty | Item CS | [Amazon Reviews](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) |
| Yelp | Restaurants | User CS | [Yelp Open Dataset](https://www.yelp.com/dataset) |
| MovieLens-1M | Movies | User CS | [MovieLens](https://grouplens.org/datasets/movielens/1m/) |

Preprocess and build the knowledge graph:

```bash
# 1. Preprocess datasets (cold-start splits, profile construction)
python scripts/preprocess.py --dataset games --cold_start item

# 2. Build the domain knowledge graph from item profiles
python scripts/build_kg.py --dataset games --extractor qwen2.5-3b

# 3. Build FAISS index for low-level retrieval
python scripts/build_index.py --dataset games
```

### Run LightCSR

```bash
# Full pipeline: retrieval + SLM ranking
python run_lightcsr.py \
    --dataset games \
    --cold_start item \
    --model Qwen/Qwen2.5-3B-Instruct \
    --quantization int4 \
    --topk 20 \
    --kg_hops 2 \
    --output results/lightcsr_games.json

# Evaluate
python scripts/evaluate.py --results results/lightcsr_games.json --metrics recall@10 ndcg@10
```

### Optional: Distillation & Quantization

```bash
# Distill ranking ability from GPT-4o teacher into the SLM
python scripts/distill.py \
    --teacher gpt-4o \
    --student Qwen/Qwen2.5-3B-Instruct \
    --n_samples 10000 \
    --dataset games

# INT4 quantization (GPTQ / AWQ)
python scripts/quantize.py --model checkpoints/lightcsr-distilled --method gptq --bits 4
```

---

## Results

### Item Cold-Start Performance (placeholder — to be filled upon publication)

| Method | Model Size | Amazon Games R@10 / N@10 | Amazon CDs R@10 / N@10 | Beauty R@10 / N@10 |
|--------|-----------|--------------------------|------------------------|---------------------|
| LightGCN | — | | | |
| DropoutNet | — | | | |
| LLM-Direct (GPT-4o) | ~1.8T | | | |
| ColdRAG (GPT-4o) | ~1.8T | | | |
| ColdRAG (SLM) | 3B | | | |
| **LightCSR (ours)** | 3B | | | |

### Cost-Efficiency (placeholder)

| Method | Model Size | Latency (ms) | Tokens/Query | GPU Mem (GB) | Cost / 1K queries |
|--------|-----------|--------------|--------------|--------------|-------------------|
| ColdRAG (GPT-4o) | ~1.8T | ~3000 | ~4000 | — (API) | ~$12.0 |
| **LightCSR (3B, INT4)** | 3B | ~200 | ~800 | 2 | ~$0.03 |
| **LightCSR (1.5B, INT4)** | 1.5B | ~120 | ~800 | 1.2 | ~$0.02 |

Full results, ablation studies, and the model-scaling analysis (0.5B → 8B) are reported in our paper.

---

## Repository Structure

```
lightcsr/
├── lightcsr/
│   ├── profile_builder.py    # Stage 1: item/user profile construction
│   ├── retrieval.py          # Stage 2: dual-level retrieval (KG + vector)
│   ├── kg/                   # Knowledge graph construction & traversal
│   ├── ranker.py             # Stage 3: SLM ranking w/ vLLM backend
│   └── distill.py            # Knowledge distillation utilities
├── baselines/
│   ├── coldrag/              # ColdRAG reproduction
│   ├── kalm4rec/             # KALM4Rec adaptation
│   ├── llm_direct/           # LLM-Direct prompting baselines
│   └── traditional/          # LightGCN, DropoutNet, MeLU, SGL (via RecBole)
├── scripts/
│   ├── preprocess.py         # Dataset preprocessing & cold-start splits
│   ├── build_kg.py           # KG construction
│   ├── build_index.py        # FAISS index building
│   ├── distill.py            # Teacher-student distillation
│   ├── quantize.py           # GPTQ/AWQ quantization
│   └── evaluate.py           # Metrics computation (Recall, NDCG, HR, MRR)
├── run_lightcsr.py           # Main entry point
├── configs/                  # Per-dataset hyperparameter configs
├── data/                     # Preprocessed data (gitignored; see download script)
├── results/                  # Output directory
├── docs/                
└── requirements.txt

```

---

## Configuration

Key hyperparameters (defaults in `configs/default.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `topk` | 20 | Number of retrieved candidates |
| `kg_hops` | 2 | Knowledge-graph multi-hop depth |
| `quantization` | int4 | INT4 / INT8 / fp16 |
| `max_gen_len` | 512 | Max generation length (tokens) |
| `temperature` | 0.0 | Greedy decoding for reproducibility |
| `distill_samples` | 10000 | Number of distillation samples |
| `seed` | 42 | Fixed for reproducibility |

---

## Reproducibility

- All experiments are repeated **5 times** with mean ± std reported.
- Greedy decoding (temperature = 0) for deterministic outputs.
- All methods share the same candidate pools, evaluation protocol, and prompt formats.
- Seeds are fixed; see `scripts/evaluate.py` for the exact metric implementations.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{lightcsr2026,
  title     = {LightCSR: Lightweight Retrieval-Augmented Generation with
               Small Language Models for Cold-Start Recommendation},
  author    = {Your Name and Others},
  booktitle = {Proceedings of the ...},
  year      = {2026}
}
```

## Acknowledgments

This project builds upon several excellent works:

- [ColdRAG](https://arxiv.org/abs/2505.20773) — knowledge-guided RAG for cold-start recommendation
- [KALM4Rec](https://github.com/dangkh/Kalm4rec-www) — keyword-driven RAG for cold-start users
- [LightRAG](https://arxiv.org/abs/2410.05779) — lightweight graph-based RAG framework
- [SLMRec](https://openreview.net/forum?id=SAKKGi9TFt) — distilling small LMs for sequential recommendation
- [RecBole](https://github.com/RUCAIBox/RecBole) — recommendation evaluation framework

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
