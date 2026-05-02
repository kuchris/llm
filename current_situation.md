# Current Situation

## What was changed and why

### `step5_attention.py` — Flash Attention

Replaced the manual attention implementation with `F.scaled_dot_product_attention` (Flash Attention path on CUDA).

**Removed:**
- `import math` (no longer needed)
- `register_buffer("causal_mask", ...)` — Flash Attention handles the causal mask internally
- `self.dropout = nn.Dropout(args.dropout)` — replaced with a plain float store

**Added:**
- `self.attn_dropout = args.dropout` — stores the dropout rate as a float
- In `forward()`, replaced the manual scores/mask/softmax/dropout block with:
  ```python
  xq = xq.transpose(1, 2)
  xk = xk.transpose(1, 2)
  xv = xv.transpose(1, 2)
  dropout_p = self.attn_dropout if self.training else 0.0
  output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=dropout_p, is_causal=True)
  output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
  return self.wo(output)
  ```

---

### `step9_train_tiny_transformer.py` — Training improvements

#### Bug fixes

1. **NameError on char tokenizer**: `task` and `text` were referenced before being defined. Fixed by moving their assignment before the tokenizer construction block.
2. **Resume off-by-one**: `start_step > max_iters` allowed one extra step after completion. Fixed to `start_step >= max_iters`.

#### Performance and training quality

3. **`import math`** added at top (needed for cosine LR schedule).

4. **cuBLAS settings** added right after device detection:
   ```python
   if device == "cuda":
       torch.backends.cudnn.benchmark = False   # prevents multi-minute hang on first step with unusual shapes (e.g. 151k vocab)
       torch.backends.cuda.matmul.allow_tf32 = True
   ```

5. **Mixed precision (BF16 AMP)**:
   ```python
   use_amp = device == "cuda"
   amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
   scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
   ```
   On RTX 4080 laptop: BF16 is supported, so `amp_dtype = bfloat16` and `scaler` is disabled (not needed for BF16).

6. **LR schedule** — warmup 500 steps then cosine decay to 10% of peak LR:
   ```python
   warmup_steps = min(500, max_iters // 20)
   def lr_lambda(step): ...
   for group in optimizer.param_groups:
       group.setdefault("initial_lr", learning_rate)  # required for LambdaLR when last_epoch >= 0
   scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=start_step - 1)
   ```

7. **`torch.compile`** with Windows guard (Triton is not available on Windows, so it is skipped):
   ```python
   if device == "cuda" and sys.platform != "win32":
       model = torch.compile(model)
   ```

8. **Training step** wrapped in `torch.autocast`, backward pass updated to use scaler, gradient clipping added:
   ```python
   with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
       logits = model(x)
       loss = ...
   optimizer.zero_grad(set_to_none=True)
   scaler.scale(loss).backward()
   scaler.unscale_(optimizer)
   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   scaler.step(optimizer)
   scaler.update()
   scheduler.step()
   ```

9. **Preset `batch_size`** for both `qwen_tokenizer_tiny_pretrain_sharded` and `qwen_tokenizer_tiny_sft` changed from `2` to `4` (manually by user, to fit 12 GB VRAM).

---

## Hardware context

- GPU: RTX 4080 laptop, 12 GB VRAM, Ada Lovelace (SM 8.9)
- BF16 supported: yes → GradScaler is disabled, autocast uses bfloat16
- torch.compile: skipped on Windows (no Triton)
- Flash Attention: works via `F.scaled_dot_product_attention` built-in CUDA kernels (no Triton needed)

## Model context

- Tokenizer: Qwen3-0.6b (HFTokenizer), vocab size ~151,936
- Model params: ~292M total (233M from embedding + output projection due to large vocab)
- The `cudnn.benchmark = False` fix was critical — without it, cuBLAS autotuning the 151k-vocab output projection caused a 4+ minute hang on the very first training step.
