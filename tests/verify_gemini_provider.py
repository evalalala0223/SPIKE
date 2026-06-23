"""Minimal end-to-end check for the Gemini backend via SPIKE's GeminiProvider.

It loads env/.env, builds the real GeminiProvider from agent/conf/gemini_config.json,
and sends exactly one completion request. Use this to confirm the Gemini API is
reachable and the key/model are configured correctly, without launching the game.

Usage:
    python verify_gemini_provider.py
    python verify_gemini_provider.py --config agent/conf/gemini_config.json --prompt "hello"
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "agent"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Gemini backend through SPIKE GeminiProvider.")
    parser.add_argument("--config", default="agent/conf/gemini_config.json", help="LLM config path (relative to repo root).")
    parser.add_argument("--prompt", default="Reply with exactly: SPIKE_GEMINI_OK", help="Prompt for the single test request.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load GEMINI_KEY (and others) from env/.env so os.getenv works.
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, "env", ".env"))
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not load env/.env automatically: {exc}", file=sys.stderr)

    cfg_path = os.path.join(ROOT, args.config)
    with open(cfg_path, "r", encoding="utf-8") as fd:
        cfg = json.load(fd)

    key_var = cfg.get("key_var", "GEMINI_KEY")
    if not os.getenv(key_var):
        print(f"[fail] env var {key_var} is empty. Fill it in env/.env first.", file=sys.stderr)
        return 2

    print(f"[info] model={cfg.get('comp_model')} | key_var={key_var} (set) | emb_model={cfg.get('emb_model')}")

    from stardojo.provider.llm.gemini import GeminiProvider

    provider = GeminiProvider()
    # Pass the dict directly to bypass project-path resolution.
    provider.init_provider(cfg)

    messages = [{"role": "user", "parts": [{"text": args.prompt}]}]
    text, info = provider.create_completion(
        messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )

    print("[response]", (text or "").strip())
    print("[tokens]", info)

    if text and text.strip():
        print("[pass] Gemini backend reachable via GeminiProvider.")
        return 0
    print("[fail] empty response from Gemini.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
