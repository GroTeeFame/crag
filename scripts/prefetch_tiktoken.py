#!/usr/bin/env python3
"""
Prefetch tiktoken encodings into a local cache directory.

Usage:
  python scripts/prefetch_tiktoken.py --dir vendor/tiktoken_cache 

Then commit the cache folder to your repo and, on the server, set:
  TIKTOKEN_CACHE_DIR=/opt/crag/app/vendor/tiktoken_cache

This avoids outbound network calls from tiktoken at runtime (useful behind
strict proxies).
"""
import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="vendor/tiktoken_cache", help="Target cache directory")
    ap.add_argument(
        "--enc",
        nargs="*",
        default=["cl100k_base", "o200k_base", "r50k_base"],
        help="Encoding names to prefetch",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=["gpt-4o", "gpt-4.1"],
        help="Model names to seed via encoding_for_model",
    )
    args = ap.parse_args()

    cache_dir = os.path.abspath(args.dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir

    try:
        import tiktoken
    except Exception as e:
        print("tiktoken is not installed in this environment:", e, file=sys.stderr)
        return 1

    print(f"Using cache dir: {cache_dir}")

    # Prefetch by explicit encoding name
    for enc_name in args.enc:
        try:
            print(f"- Fetching encoding: {enc_name} …", end="", flush=True)
            _ = tiktoken.get_encoding(enc_name)
            print(" ok")
        except Exception as e:
            print(f" failed: {e}")

    # Seed by model name as well (resolves to an encoding under the hood)
    for model in args.models:
        try:
            print(f"- Seeding encoding_for_model: {model} …", end="", flush=True)
            _ = tiktoken.encoding_for_model(model)
            print(" ok")
        except Exception as e:
            print(f" failed: {e}")

    print("Done. You can now commit the cache directory:")
    print(f"  git add {args.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

