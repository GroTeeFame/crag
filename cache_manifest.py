import os
import json
import time
import hashlib
from typing import Optional, Dict, Any


MANIFEST_PATH = os.getenv("PROCESSED_MANIFEST", "work_results/processed_manifest.json")


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def _load(path: str = MANIFEST_PATH) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _save(data: Dict[str, Any], path: str = MANIFEST_PATH) -> None:
    _ensure_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def sha256_file(path: str, bufsize: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def cache_lookup(input_hash: str) -> Optional[str]:
    """Return output file path for a previously processed input hash, if present and file exists."""
    data = _load()
    rec = data.get(input_hash)
    if not rec:
        return None
    out = rec.get("output_path")
    if out and os.path.exists(out):
        return out
    return None


def cache_store(input_hash: str, output_path: str, meta: Optional[Dict[str, Any]] = None) -> None:
    data = _load()
    data[input_hash] = {
        "output_path": output_path,
        "finished_at": int(time.time()),
        **(meta or {}),
    }
    _save(data)

