# Tiny LLM Lessons

This repo is a learning project for building and testing small language-model pieces in PyTorch.

The repo now has two different tracks:

```text
1. Scratch TinyTransformer
   Learn tokenizers, RoPE, grouped-query attention, MLPs, decoder layers,
   training loops, checkpoints, and generation.

2. Local Qwen3 0.6B
   Download and run a real pretrained small model for comparison.
```

The important conclusion from the current experiments is simple:

```text
The scratch model is useful for learning how transformers work.
It is not strong enough to behave like a real assistant.
For useful answers, use pretrained Qwen weights and fine-tune later.
```

## Current Situation

The scratch model can train and generate, but quality is poor for general conversation. It learns format and repeated patterns, but it does not have enough data, compute, or training time to learn broad knowledge.

The local Qwen model works much better immediately because the knowledge is already in its pretrained weights:

```text
models/qwen3-0.6b/
```

That folder is ignored by git because the weights are large.

## Environment

Use Python 3.12 with `uv`:

```powershell
uv sync --python 3.12
```

For VS Code notebooks:

```powershell
uv add --dev ipykernel
```

On Windows, the project is configured to install CUDA PyTorch from the PyTorch `cu128` index through `pyproject.toml`.

## Main Files

Core scratch model:

```text
step1_tokenizer.py                 character tokenizer
step2_training_samples.py          next-token sample builder
step3_tiny_llama_parts.py          ModelConfig and RMSNorm
step4_rope.py                      RoPE positional encoding
step5_attention.py                 causal grouped-query attention
step6_mlp.py                       LLaMA-style SwiGLU MLP
step7_decoder_layer.py             one decoder block
step8_tiny_transformer.py          full scratch Transformer
step9_train_tiny_transformer.py    scratch training script
step10_generate.py                 scratch generation script
```

Data and tokenizer prep:

```text
download/step11_download_tiny_shakespeare.py
download/step12_download_wikitext2.py       downloads WikiText-103 now
step13_bpe_tokenizer.py                     BPE tokenizer trainer
step15_prepare_wikitext2.py                 WikiText-103 cleaner
step16_prepare_alpaca.py                    Alpaca formatter
step20_prepare_dolly.py                     Dolly-15k formatter
step21_prepare_bea_grammar.py               BEA / WI+LOCNESS grammar formatter
step22_prepare_free_tokenizer_train.py      builds combined tokenizer text
```

Qwen tools:

```text
download/step23_download_qwen3.py     downloads Qwen/Qwen3-0.6B
step24_generate_qwen3.py              runs Qwen weights + Qwen tokenizer
step25_inspect_qwen3.py               inspects Qwen config, tokenizer, weights
step26_compare_tokenizers.py          compares repo BPE vs Qwen tokenizer
step27_test_qwen3.py                  batch tests Qwen prompts
step28_estimate_scratch_model.py      estimates scratch model size/memory
hf_tokenizer.py                       wrapper for Hugging Face tokenizers
```

Notebooks:

```text
colab_tiny_llm.ipynb                  local VS Code scratch/free-data pipeline
qwen3_0_6b_local.ipynb                pretrained Qwen test notebook
qwen_tokenizer_scratch_model.ipynb    scratch model using Qwen tokenizer only
```

## Scratch Model Config

The default scratch model shape is in `step3_tiny_llama_parts.py`:

```python
dim: int = 768
n_layers: int = 12
n_heads: int = 16
n_kv_heads: int = 8
head_dim: int | None = None
vocab_size: int = 6144
hidden_dim: int | None = None
multiple_of: int = 64
max_seq_len: int = 512
norm_eps: float = 1e-5
dropout: float = 0.1
```

If `head_dim` is `None`, attention uses:

```text
head_dim = dim // n_heads
```

Qwen-style experiments can set `head_dim=128`, where attention projects to `n_heads * head_dim` instead of just `dim`.

## Scratch Training Presets

The active scratch training presets live in `step9_train_tiny_transformer.py`.

Current default:

```text
free_wikitext103_pretrain_bpe
```

Free-data path:

```text
free_wikitext103_pretrain_bpe
free_dolly_sft_bpe
free_bea_grammar_sft_bpe
```

Qwen-tokenizer scratch path:

```text
qwen_tokenizer_tiny_pretrain
qwen_tokenizer_tiny_sft
```

Qwen-shape scratch experiment:

```text
qwen_shape_scratch_pretrain
```

The Qwen-shape experiment is not recommended on this machine. The estimator reports about `751M` parameters and roughly `11 GiB` of AdamW training memory before activations, so it is too close to the limit of a 12 GB laptop GPU.

## Build Free Scratch Data

This is the current free-data scratch pipeline:

```powershell
uv run python download/step12_download_wikitext2.py
uv run python step15_prepare_wikitext2.py
uv run python step20_prepare_dolly.py
uv run python step21_prepare_bea_grammar.py
uv run python step22_prepare_free_tokenizer_train.py
```

Train the shared `6144` BPE tokenizer:

```powershell
uv run python step13_bpe_tokenizer.py --vocab-size 6144 --max-chars 10000000 --special-token '<unk>' --special-token '<eos>'
```

This writes:

```text
tokenizers/free_bpe_6144.json
```

## Train Scratch Model

Pretrain:

```powershell
uv run python step9_train_tiny_transformer.py --preset free_wikitext103_pretrain_bpe --device auto
```

General SFT:

```powershell
uv run python step9_train_tiny_transformer.py --preset free_dolly_sft_bpe --device auto
```

Grammar SFT:

```powershell
uv run python step9_train_tiny_transformer.py --preset free_bea_grammar_sft_bpe --device auto
```

Resume a stopped run:

```powershell
uv run python step9_train_tiny_transformer.py --preset free_wikitext103_pretrain_bpe --device auto --resume
```

Autosave defaults to every `200` steps:

```powershell
uv run python step9_train_tiny_transformer.py --preset free_wikitext103_pretrain_bpe --device auto --save-every 50
```

Checkpoint files are not updated continuously. They are written only on autosave, interrupt, or final save.

## Generate With Scratch Model

After SFT:

```powershell
uv run python step10_generate.py --preset free_dolly_sft_bpe --device auto --prompt "Where is Hong Kong?"
```

For the BEA grammar stage:

```powershell
uv run python step10_generate.py --preset free_bea_grammar_sft_bpe --device auto --prompt "Correct this sentence: She go to school yesterday."
```

Current limitation:

```text
The scratch model output is often bad.
This is expected for a small model trained from scratch on limited data.
```

## Run Qwen3 0.6B

Download:

```powershell
uv run python download/step23_download_qwen3.py
```

Generate:

```powershell
uv run python step24_generate_qwen3.py --device auto --prompt "Where is Hong Kong?"
```

Batch test:

```powershell
uv run python step27_test_qwen3.py --device auto
```

Qwen uses:

```text
Qwen tokenizer
Qwen pretrained weights
```

That is why it can answer much better than the scratch model.

## Inspect Qwen

Inspect architecture, tokenizer, tensor shapes, and parameter counts:

```powershell
uv run python step25_inspect_qwen3.py
```

Useful options:

```powershell
uv run python step25_inspect_qwen3.py --limit 50
uv run python step25_inspect_qwen3.py --raw-config
```

The downloaded `Qwen/Qwen3-0.6B` config is roughly:

```text
dim / hidden_size: 1024
n_layers: 28
n_heads: 16
n_kv_heads: 8
head_dim: 128
vocab_size: 151936
hidden_dim / intermediate_size: 3072
max_position_embeddings: 40960
norm_eps: 1e-6
dropout: 0.0
```

The local safetensors count is about:

```text
total tensors: 751.6M parameters
model body:    596.0M parameters
lm_head:       155.6M parameters
```

## Tokenizer Experiments

Compare repo BPE with Qwen tokenizer:

```powershell
uv run python step26_compare_tokenizers.py
```

Custom text:

```powershell
uv run python step26_compare_tokenizers.py --text "Where is Hong Kong?"
```

Qwen tokenizer is much more efficient, especially for multilingual text and code. But using Qwen tokenizer with the scratch model does not transfer Qwen knowledge.

This experiment is available:

```powershell
uv run python step9_train_tiny_transformer.py --preset qwen_tokenizer_tiny_pretrain --device auto
uv run python step9_train_tiny_transformer.py --preset qwen_tokenizer_tiny_sft --device auto
```

It is mostly educational. Checkpoints are large because the Qwen vocabulary is about `151k`.

## What We Tried

These experiments were useful but not the final path:

```text
WikiText-103 + Dolly + BEA from scratch
Qwen tokenizer with scratch weights
Qwen-shaped scratch model
```

Observed results:

```text
The scratch model learns format faster than facts.
BEA grammar SFT biases the model toward essay-like rewriting.
Qwen tokenizer improves tokenization but not knowledge.
Qwen-shaped scratch training is too heavy for the current GPU setup.
```

## Practical Next Step

The useful direction is:

```text
Use Qwen pretrained weights
Collect or prepare custom instruction data
Fine-tune with LoRA / QLoRA
Evaluate before replacing the model
```

Do not train on the model's own unverified outputs. A self-improving loop needs human corrections, filtering, evaluation, and checkpoint promotion.

## Artifact Policy

These folders are generated and ignored by git:

```text
checkpoints/
data/
models/
tokenizers/
results/
```

That keeps large datasets, weights, and checkpoints out of the repository.
