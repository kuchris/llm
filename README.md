# Tiny LLM Lessons

This repo is a step-by-step learning project for building a tiny LLaMA-style language model in PyTorch.

The goal is to understand the machine: tokenizers, embeddings, RoPE, attention, MLPs, decoder layers, training, checkpoints, and generation. This is not meant to compete with GPT/Qwen/Llama models.

## Current Main Path

The current recommended path is:

```text
clean WikiText pretraining
-> assistant SFT with response-only loss
-> <eos>-based generation stop
```

Default training preset:

```python
DEFAULT_PRESET = "assistant_eos_sft_bpe"
```

Default generation preset:

```python
DEFAULT_PRESET = "assistant_eos_sft_bpe"
```

## Requirements

Use `uv`:

```powershell
uv run python step10_generate.py
```

The project pins Python to 3.12 because PyTorch is more stable there.

## File Map

```text
step1_tokenizer.py                    character tokenizer
step2_training_samples.py             next-token sample builder
step3_tiny_llama_parts.py             ModelConfig and RMSNorm
step4_rope.py                         RoPE positional encoding
step5_attention.py                    causal grouped-query attention
step6_mlp.py                          LLaMA-style MLP / SwiGLU
step7_decoder_layer.py                one decoder layer
step8_tiny_transformer.py             full tiny Transformer
step9_train_tiny_transformer.py       training script
step10_generate.py                    generation script
download/step11_download_tiny_shakespeare.py   Tiny Shakespeare downloader
download/step12_download_wikitext2.py          WikiText-2 downloader
step13_bpe_tokenizer.py               BPE tokenizer trainer/wrapper
step15_prepare_wikitext2.py           WikiText cleaner
step16_prepare_alpaca.py              Alpaca SFT formatter
step17_prepare_assistant_sft.py       assistant identity SFT data
step18_prepare_assistant_tokenizer_train.py
step19_prepare_assistant_eos.py       assistant SFT data with <eos>
```

## Data Files

```text
data/tiny_text.txt
data/tiny_shakespeare.txt
data/wikitext2.txt
data/wikitext2_clean.txt
data/alpaca_sft.txt
data/assistant_sft.txt
data/assistant_tokenizer_train.txt
data/assistant_eos_sft.txt
data/assistant_eos_tokenizer_train.txt
```

## Tokenizers

Character tokenizer:

```text
text -> one token per unique character
```

BPE tokenizer:

```text
text -> learned chunks/subwords
```

The current EOS assistant tokenizer is:

```text
tokenizers/assistant_eos_bpe_4000.json
```

It includes:

```text
<unk>
<eos>
```

`<eos>` means end of response. Generation stops when the model produces it.

## Build Data

Download WikiText-2:

```powershell
uv run python download/step12_download_wikitext2.py
```

Clean WikiText:

```powershell
uv run python step15_prepare_wikitext2.py
```

Prepare Alpaca:

```powershell
uv run python step16_prepare_alpaca.py
```

Prepare assistant EOS data:

```powershell
uv run python step19_prepare_assistant_eos.py
```

Train EOS BPE tokenizer:

```powershell
uv run python step13_bpe_tokenizer.py --data data/assistant_eos_tokenizer_train.txt --out tokenizers/assistant_eos_bpe_4000.json --vocab-size 4000 --max-chars 3000000 --special-token '<unk>' --special-token '<eos>'
```

## Train

Pretrain first:

```powershell
uv run python step9_train_tiny_transformer.py --preset assistant_eos_pretrain_bpe
```

Then SFT:

```powershell
uv run python step9_train_tiny_transformer.py --preset assistant_eos_sft_bpe
```

The SFT preset loads the pretrain checkpoint:

```text
checkpoints/assistant_eos_pretrain_bpe/tiny_transformer.pt
```

and saves:

```text
checkpoints/assistant_eos_sft_bpe/tiny_transformer.pt
```

## Generate

After SFT:

```powershell
uv run python step10_generate.py
```

Example prompt:

```text
hi, who are you?
```

The script wraps it as:

```text
### Instruction:
hi, who are you?

### Response:
```

Then it stops at `<eos>`.

## Key Concepts

Embedding:

```text
token id -> learned vector
```

Attention:

```text
tokens look at useful previous tokens
```

MLP:

```text
each token vector is transformed independently
```

RoPE:

```text
adds position information by rotating query/key vectors
```

Pretraining:

```text
learn general next-token prediction from normal text
```

SFT:

```text
learn instruction -> response behavior
```

Response-only loss:

```text
only response tokens count toward SFT loss
```

## Model Size

Model shape is in `step3_tiny_llama_parts.py`:

```python
dim: int = 128
n_layers: int = 4
n_heads: int = 4
n_kv_heads: int = 2
```

If you change model shape, train a new checkpoint. Old checkpoints with different shapes will not load.

## Current Limits

This is still a tiny CPU-trained model. It can learn format and repeated behaviors, but it may hallucinate and fail broad reasoning tasks.

For real assistant quality, use a pretrained model and fine-tune it. This project is for learning how the pieces work.
