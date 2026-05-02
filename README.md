# Tiny LLM Lessons

This repo is a PyTorch learning project for building a small decoder-only language model from scratch.

The current active setup is:

```text
Qwen tokenizer
scratch TinyTransformer weights
FineWeb-Edu pretraining shards
Alpaca-cleaned SFT
CUDA training on Windows
```

Important: using the Qwen tokenizer does **not** transfer Qwen's pretrained knowledge. It only gives better tokenization. For reliable factual question answering, use pretrained Qwen weights and fine-tune with LoRA/QLoRA. The scratch path is mainly for learning the training pipeline.

## Environment

Install dependencies:

```powershell
uv sync --python 3.12
```

For VS Code notebooks:

```powershell
uv add --dev ipykernel
```

The project uses CUDA PyTorch from the PyTorch `cu128` index configured in `pyproject.toml`.

## Main Workflow

### 1. Download Qwen3 0.6B

```powershell
uv run python download/step23_download_qwen3.py
```

This creates:

```text
models/qwen3-0.6b/
```

### 2. Prepare FineWeb-Edu Text

The main pretraining text file is:

```text
data/fineweb_edu_sample_10bt.txt
```

If this full file already exists, do **not** rerun `step29_prepare_fineweb_edu.py`; it opens the output path for writing and can overwrite the full file.

### 3. Tokenize FineWeb-Edu Into Shards

For large pretraining, use token shards instead of loading the full text file into RAM:

```powershell
uv run python step32_tokenize_pretrain_shards.py
```

This writes:

```text
data/fineweb_edu_qwen_uint32/manifest.json
data/fineweb_edu_qwen_uint32/shard_*.bin
```

Qwen token ids require `uint32`, not `uint16`, because token ids can exceed `65535`.

If tokenization is stopped early, finalize the existing shards:

```powershell
uv run python step32_tokenize_pretrain_shards.py --finalize-existing
```

### 4. Pretrain From Shards

```powershell
uv run python step9_train_tiny_transformer.py --preset qwen_tokenizer_tiny_pretrain_sharded --device auto --resume --save-every 10000
```

Current sharded pretrain preset:

```text
dim=768
n_layers=8
n_heads=8
n_kv_heads=4
block_size=256
batch_size=4
tokenizer=models/qwen3-0.6b
checkpoint=checkpoints/qwen_tokenizer_tiny_pretrain/tiny_transformer.pt
```

### 5. Prepare Alpaca-Cleaned SFT

```powershell
uv run python step30_prepare_alpaca_cleaned.py
```

This writes:

```text
data/alpaca_cleaned_train_eos_sft.txt
data/alpaca_cleaned_eval_eos_sft.txt
```

### 6. Run SFT

```powershell
uv run python step9_train_tiny_transformer.py --preset qwen_tokenizer_tiny_sft --device auto --resume --save-every 1000
```

Current SFT preset starts from:

```text
checkpoints/qwen_tokenizer_tiny_pretrain/tiny_transformer.pt
```

and writes:

```text
checkpoints/qwen_tokenizer_tiny_alpaca_sft/tiny_transformer.pt
```

### 7. Generate

From the raw pretrain checkpoint:

```powershell
uv run python step10_generate.py --preset qwen_tokenizer_tiny_pretrain --device auto --prompt "Hong Kong is" --max-new-tokens 80 --temperature 0.3
```

After SFT:

```powershell
uv run python step10_generate.py --preset qwen_tokenizer_tiny_sft --device auto --prompt "How are you?" --max-new-tokens 80 --temperature 0.3
```

## Training Implementation

The current trainer uses:

```text
memmap token shards for large pretraining data
PyTorch scaled_dot_product_attention with causal masking
BF16 autocast on CUDA when supported
TF32 matmul enabled
warmup + cosine learning-rate schedule
gradient clipping
checkpoint resume
```

Checkpoint files are saved on autosave, interrupt, or final completion. They are not updated every step.

## Quality Expectation

10k pretrain steps is only a pipeline test. Bad or repetitive generation at that point is expected.

The scratch model needs much longer pretraining plus SFT before it can follow instructions at all, and it still will not be a reliable factual QA model. For useful answers, run or fine-tune pretrained Qwen weights instead.

## Run Pretrained Qwen For Comparison

Generate with real pretrained Qwen weights:

```powershell
uv run python step24_generate_qwen3.py --device auto --prompt "Where is Hong Kong?"
```

Batch test:

```powershell
uv run python step27_test_qwen3.py --device auto
```

Inspect model/tokenizer details:

```powershell
uv run python step25_inspect_qwen3.py
```

## Important Files

Core model:

```text
step3_tiny_llama_parts.py       ModelConfig and RMSNorm
step4_rope.py                   RoPE
step5_attention.py              grouped-query attention with SDPA
step6_mlp.py                    SwiGLU MLP
step7_decoder_layer.py          decoder block
step8_tiny_transformer.py       full Transformer
step9_train_tiny_transformer.py training
step10_generate.py              generation
```

Current data/tokenizer tools:

```text
hf_tokenizer.py                       Hugging Face tokenizer wrapper
step29_prepare_fineweb_edu.py         FineWeb-Edu text preparation
step30_prepare_alpaca_cleaned.py      Alpaca-cleaned SFT preparation
step32_tokenize_pretrain_shards.py    FineWeb-Edu Qwen-token shard preparation
step26_compare_tokenizers.py          tokenizer comparison
```

Pretrained Qwen tools:

```text
download/step23_download_qwen3.py
step24_generate_qwen3.py
step25_inspect_qwen3.py
step27_test_qwen3.py
```

Main notebook:

```text
qwen_tokenizer_scratch_model.ipynb
```

## Generated Files

These folders are generated and ignored by git:

```text
checkpoints/
data/
models/
tokenizers/
results/
```
