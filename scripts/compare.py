"""Greedy continuations from several models for each prompt, printed one prompt at a time.

  .venv/bin/python scripts/compare.py                          # prompts.txt, base vs both fp32 models
  .venv/bin/python scripts/compare.py --models base mlx-bf16 torch-bf16 --max-tokens 200
  .venv/bin/python scripts/compare.py --prompt "In this paper we prove"

Model names: base, or a run name (mlx-fp32, torch-fp32, mlx-bf16, torch-bf16, mlx-fp32-tf32),
which resolves to its converted checkpoint in checkpoints/<run>-mlx.
These are base (not chat) models: give them text to continue, not questions.
"""

import argparse
import textwrap

import mlx.core as mx
import mlx_lm

import common as C


def model_path(name):
    return C.BASE_MODEL if name == "base" else C.CKPT / f"{name}-mlx"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["base", "mlx-fp32", "torch-fp32"])
    p.add_argument("--prompts-file", default=str(C.ROOT / "prompts.txt"))
    p.add_argument("--prompt", help="a single prompt instead of the prompts file")
    p.add_argument("--max-tokens", type=int, default=150)
    args = p.parse_args()

    if args.prompt:
        prompts = [args.prompt]
    else:
        # One prompt per line; a literal \n inside a line becomes a newline.
        lines = open(args.prompts_file).read().splitlines()
        prompts = [l.replace("\\n", "\n") for l in lines if l.strip() and not l.startswith("#")]

    outputs = {}
    for name in args.models:
        model, tokenizer = mlx_lm.load(str(model_path(name)))
        outputs[name] = [continue_text(model, tokenizer, prompt, args.max_tokens) for prompt in prompts]
        del model
        mx.clear_cache()

    for i, prompt in enumerate(prompts):
        print("=" * 100)
        print(f"PROMPT: {prompt!r}")
        for name in args.models:
            print(f"\n--- {name}")
            print(textwrap.fill(outputs[name][i], width=100, replace_whitespace=False))
        print()


def continue_text(model, tokenizer, prompt, max_tokens):
    """Greedy continuation of the raw prompt (no chat template), returned with the prompt.

    Decodes prompt and continuation together: decoding the continuation alone drops the
    leading space of its first word.
    """
    prompt_ids = tokenizer.encode(prompt)
    new_ids = [r.token for r in mlx_lm.stream_generate(model, tokenizer, prompt_ids, max_tokens=max_tokens)]
    return tokenizer.decode(prompt_ids + new_ids)


if __name__ == "__main__":
    main()
