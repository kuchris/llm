import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = "models/qwen3-0.6b"
DEFAULT_PROMPTS = [
    "Where is Hong Kong?",
    "Correct this sentence: She go to school yesterday.",
    "Identify the mistakes in this sentence: The house was painted green and purple.",
    "Explain in one paragraph why a tiny model trained from scratch gives bad answers.",
]


def build_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return prompt


def generate_one(
    *,
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    model_input = build_prompt(tokenizer, prompt)
    inputs = tokenizer(model_input, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model directory not found: {model_path}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch install has no CUDA support.")

    print(f"device: {device}")
    print(f"model: {model_path}")
    print("tokenizer: Qwen AutoTokenizer")
    print("weights: Qwen AutoModelForCausalLM")
    print()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    prompts = args.prompt or DEFAULT_PROMPTS
    for index, prompt in enumerate(prompts, start=1):
        print(f"== Test {index} ==")
        print(f"prompt: {prompt}")
        answer = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(answer)
        print()


if __name__ == "__main__":
    main()
