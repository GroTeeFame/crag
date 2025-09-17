#!/usr/bin/env python3
"""
Delete files in docx_results/ older than N days (default 30).

Usage:
  python scripts/cleanup_results.py [--days 30] [--root docx_results]

Recommended to run from cron or a systemd timer once per day.
"""
import os
import time
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=30, help='Age threshold in days')
    ap.add_argument('--root', default='docx_results', help='Results directory')
    args = ap.parse_args()

    root = args.root
    days = args.days
    if not os.path.isdir(root):
        return 0

    now = time.time()
    cutoff = now - (days * 86400)
    removed = 0

    for name in os.listdir(root):
        if not name.lower().endswith('.docx'):
            continue
        p = os.path.join(root, name)
        try:
            st = os.stat(p)
        except FileNotFoundError:
            continue
        if st.st_mtime < cutoff:
            try:
                os.remove(p)
                removed += 1
            except Exception:
                pass

    print(f"Removed {removed} file(s) older than {days} days from {root}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

