# Tiny LLM From Scratch

## Short Pitch

This project is a hands-on learning build of a tiny LLaMA-style language model in PyTorch.

Instead of only calling an existing AI API, this project builds the important parts of a language model directly:

```text
tokenizer -> embeddings -> RoPE -> attention -> MLP -> decoder layers -> training -> generation
```

The goal is not to beat modern LLMs. The goal is to understand how they work internally.

## Why I Built This

Most people use LLMs from the outside:

```text
prompt in -> answer out
```

This project looks inside the box:

```text
text becomes token ids
token ids become vectors
vectors pass through Transformer layers
the model predicts the next token
training updates the weights
```

By building each step manually, I learned what an LLM is doing at a mechanical level.

## What The Project Can Do

The project can:

- train a character-level tokenizer
- train a custom BPE tokenizer
- build a tiny Transformer model from scratch
- pretrain on WikiText-style text
- fine-tune on instruction/response data
- use response-only SFT loss
- use `<eos>` to stop generation
- save and load PyTorch checkpoints
- generate text from a prompt

Example assistant-style prompt:

```text
hi, who are you?
```

Expected learned behavior after SFT:

```text
I am a tiny AI assistant built for learning how language models work. How can I help you?
```

## Current Architecture

The model is a small decoder-only Transformer.

Core components:

```text
Token Embedding
RoPE positional encoding
Causal self-attention
Grouped-query attention
RMSNorm
SwiGLU-style MLP
Decoder layers
Output logits
```

Model size is configured in:

```text
step3_tiny_llama_parts.py
```

Example config:

```python
dim = 128
n_layers = 4
n_heads = 4
n_kv_heads = 2
```

## Training Pipeline

The current main training pipeline has two stages.

### 1. Pretraining

Pretraining teaches the model general next-token prediction.

Dataset:

```text
data/wikitext2_clean.txt
```

Command:

```powershell
uv run python step9_train_tiny_transformer.py --preset assistant_eos_pretrain_bpe
```

Output:

```text
checkpoints/assistant_eos_pretrain_bpe/tiny_transformer.pt
```

### 2. Supervised Fine-Tuning

SFT teaches instruction-following behavior.

Dataset:

```text
data/assistant_eos_sft.txt
```

Command:

```powershell
uv run python step9_train_tiny_transformer.py --preset assistant_eos_sft_bpe
```

Output:

```text
checkpoints/assistant_eos_sft_bpe/tiny_transformer.pt
```

## Key Improvement: Response-Only Loss

In normal training, the model learns from every token.

For instruction tuning, that wastes effort on prompt text like:

```text
### Instruction:
### Response:
```

This project adds response-only SFT loss.

That means the model only learns from the answer part:

```text
### Instruction:
Who are you?

### Response:
I am a tiny AI assistant.
```

Only the response tokens count toward the SFT loss.

## Key Improvement: EOS Token

The project now uses:

```text
<eos>
```

to mark the end of a response.

This is better than manually stopping on blank lines or the next instruction marker.

Training example:

```text
### Instruction:
hi

### Response:
Hi, I am a tiny AI assistant. How can I help you?<eos>
```

During generation, the model stops when it produces `<eos>`.

## Tokenizer

The current main tokenizer is:

```text
tokenizers/assistant_eos_bpe_4000.json
```

It is a custom BPE tokenizer trained on:

```text
clean WikiText
assistant SFT data
```

Special tokens:

```text
<unk>
<eos>
```

This tokenizer is smaller and more CPU-friendly than a large GPT-2 or LLaMA tokenizer.

## What I Learned

Important lessons from this project:

- token IDs are not meanings by themselves
- embeddings are learned vector rows
- attention compares query/key vectors
- RoPE adds position information inside attention
- MLP transforms each token vector after attention
- loss is next-token prediction error
- BPE tokenization improves word structure
- pretraining and SFT solve different problems
- SFT without pretraining is weak
- small models can memorize patterns but struggle with broad knowledge
- dataset formatting strongly affects output behavior

## Limitations

This is still a tiny CPU-trained model.

It can learn:

```text
format
simple repeated behaviors
basic text rhythm
some instruction patterns
```

It cannot reliably:

```text
answer factual questions
reason like a large model
compete with pretrained LLMs
handle broad real-world tasks
```

The project is best presented as an educational LLM implementation, not a production chatbot.

## My Opinion

This is a strong learning project.

The most impressive part is not the generated text quality. The impressive part is that the project builds the full path:

```text
raw text -> tokenizer -> model -> training -> checkpoint -> SFT -> generation
```

That shows real understanding of the LLM training pipeline.

For a portfolio, I would present this as:

```text
"I implemented a tiny decoder-only language model from scratch, including BPE tokenization, RoPE attention, pretraining, response-only SFT, and EOS-based generation."
```

That is a clear and honest engineering story.

## Next Steps

Good next improvements:

- add true resume training with optimizer state
- add periodic checkpoints
- add validation loss
- add a focused reasoning dataset
- add a small evaluation script
- try LoRA fine-tuning on a pretrained model
- add GPU/Colab support

The best next portfolio project after this is LoRA fine-tuning, because it connects this low-level understanding to practical modern AI engineering.
