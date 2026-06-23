"""Minimal end-to-end check for the qwen backend via SPIKE's OpenAIProvider.

It loads env/.env, builds the real OpenAIProvider (is_opensource=True) from
agent/conf/qwen_config.json exactly like LLMFactory does for the qwen channel,
and sends exactly one completion request. Use this to confirm the qwen API
(Venus proxy) is reachable and the key/model/extra_body are configured
correctly, without launching the game.

Usage:
    python verify_qwen_provider.py
    python verify_qwen_provider.py --config agent/conf/qwen_config.json --prompt "hello"
"""

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The repo root is the dir that contains `agent/`; this script lives in tests/.
ROOT = _SCRIPT_DIR if os.path.isdir(os.path.join(_SCRIPT_DIR, "agent")) else os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT, "agent"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify qwen backend through SPIKE OpenAIProvider.")
    parser.add_argument("--config", default="agent/conf/qwen_config.json", help="LLM config path (relative to repo root).")
    parser.add_argument("--prompt", default="Reply with exactly: SPIKE_QWEN_OK", help="Prompt for the single test request.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load any keys (e.g. VENUS_QWEN_TOKEN) from env/.env so os.getenv works.
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, "env", ".env"))
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not load env/.env automatically: {exc}", file=sys.stderr)

    cfg_path = os.path.join(ROOT, args.config)
    with open(cfg_path, "r", encoding="utf-8") as fd:
        cfg = json.load(fd)

    # The qwen channel resolves api_key from the `api_key` field first, then key_var env.
    key_var = cfg.get("key_var", "")
    has_direct_key = bool(str(cfg.get("api_key", "")).strip())
    if not has_direct_key and key_var and not os.getenv(key_var):
        print(f"[fail] no api_key in config and env var {key_var} is empty.", file=sys.stderr)
        return 2

    print(
        f"[info] model={cfg.get('comp_model')} | base_url={cfg.get('base_url')} | "
        f"key=({'config.api_key' if has_direct_key else key_var + ' env'}) | "
        f"emb_model={cfg.get('emb_model')}"
    )

    from stardojo.provider.llm.openai import OpenAIProvider

    # Build exactly like LLMFactory does for the qwen channel (completion only here).
    provider = OpenAIProvider(is_opensource=True)
    provider.init_provider(cfg)  # pass dict directly to bypass project-path resolution

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": args.prompt},
    ]
    text, info = provider.create_completion(
        messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )

    print("[response]", (text or "").strip())
    print("[tokens]", info)

    if text and text.strip():
        print("[pass] qwen backend reachable via OpenAIProvider (SPIKE qwen channel).")
        return 0
    print("[fail] empty response from qwen.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
