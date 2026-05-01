# Run This Project On Google Colab

Colab can be faster than CPU if the runtime gets a GPU.

## 1. Start A GPU Runtime

In Colab:

```text
Runtime -> Change runtime type -> GPU
```

Check GPU:

```python
!nvidia-smi
```

## 2. Get The Project Into Colab

Option A: clone your GitHub repo:

```python
!git clone YOUR_REPO_URL llm
%cd llm
```

Option B: upload this folder manually, then:

```python
%cd /content/llm
```

## 3. Install uv

```python
!pip install uv
```

If you previously ran this repo in Colab before the CUDA fix, remove the old virtual environment:

```python
!rm -rf .venv
```

Then `uv run ...` will recreate `.venv` using the CUDA-capable PyTorch wheel available for Colab.

## 4. Prepare Data

Download WikiText-103:

```python
!uv run python download/step12_download_wikitext2.py
```

Clean WikiText:

```python
!uv run python step15_prepare_wikitext2.py
```

Prepare Alpaca:

```python
!uv run python step16_prepare_alpaca.py
```

Prepare EOS assistant data:

```python
!uv run python step19_prepare_assistant_eos.py
```

Train EOS tokenizer:

```python
!uv run python step13_bpe_tokenizer.py --data data/assistant_eos_tokenizer_train.txt --out tokenizers/assistant_eos_bpe_4000.json --vocab-size 4000 --max-chars 3000000 --special-token '<unk>' --special-token '<eos>'
```

## 5. Pretrain

```python
!uv run python step9_train_tiny_transformer.py --preset assistant_eos_pretrain_bpe --device cuda
```

Output:

```text
checkpoints/assistant_eos_pretrain_bpe/tiny_transformer.pt
```

## 6. SFT

```python
!uv run python step9_train_tiny_transformer.py --preset assistant_eos_sft_bpe --device cuda
```

Output:

```text
checkpoints/assistant_eos_sft_bpe/tiny_transformer.pt
```

## 7. Generate

```python
!uv run python step10_generate.py --device cuda --prompt "hi, who are you?"
```

## 8. Save To Google Drive

Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Copy checkpoints:

```python
!mkdir -p /content/drive/MyDrive/tiny_llm_checkpoints
!cp -r checkpoints /content/drive/MyDrive/tiny_llm_checkpoints/
!cp -r tokenizers /content/drive/MyDrive/tiny_llm_checkpoints/
```

## Notes

- Free Colab GPU is not guaranteed.
- Sessions can disconnect.
- Save checkpoints to Drive after long runs.
- This project now uses `--device auto` by default, so `cuda` is used when available.
