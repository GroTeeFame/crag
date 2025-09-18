#!/usr/bin/env python3
"""
Sync the Chroma vector store with documents/ based on file content hashes.

- Computes SHA-256 for every .docx under Config.DOCS_PATH (recursively)
- Compares with a manifest JSON (default: vectorstore/<NAME_FOR_DB>_manifest.json)
  to find new/modified/deleted documents
- For new/modified: upserts the single IRD via rag_logic.upsert_ird_document()
  (converts to MD, chunks, replaces vectors by attachment_group_id)
- For deleted: removes vectors by attachment_group_id and drops from manifest

Usage:
  # Preview changes only (no writes)
  python scripts/sync_vectorstore.py --dry-run

  # Initialize a baseline manifest from current files (no upserts/deletes)
  python scripts/sync_vectorstore.py --init-manifest

  # Apply changes (upsert new/changed, delete removed)
  python scripts/sync_vectorstore.py

You can schedule this daily via systemd timer (see deploy/crag-sync.*.example).
"""
from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import argparse
from typing import Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()


# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the parent directory
parent_dir = os.path.join(current_dir, '..')
# Add the parent directory to sys.path
sys.path.append(parent_dir)


from config import Config
from rag_logic import upsert_ird_document
# Prefer a bundled modern SQLite (pysqlite3) if system sqlite3 is too old
try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

from langchain_chroma import Chroma
from chromadb import PersistentClient
from rag_logic import get_embedding_model


def _default_manifest_path() -> str:
    db_dir = getattr(Config, 'DB_DIR', 'vectorstore')
    name = getattr(Config, 'NAME_FOR_DB', 'default')
    return os.path.join(db_dir, f"{name}_manifest.json")


def sha256_file(path: str, bufsize: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_relpath(docx_path: str) -> str:
    root = Config.DOCS_PATH
    return os.path.relpath(os.path.abspath(docx_path), os.path.abspath(root))


def load_manifest(path: str) -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_manifest(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def compute_attachment_group_id(docx_path: str) -> str:
    """Content-hash based group id: sha256(file contents). Fallback to path-hash if needed."""
    import hashlib
    # Try content hash first
    try:
        h = hashlib.sha256()
        with open(docx_path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        pass
    # Fallback to path-based hash
    rel = os.path.relpath(docx_path, Config.DOCS_PATH)
    rel_no_ext = os.path.splitext(rel.replace("\\", "/"))[0]
    return hashlib.sha1(rel_no_ext.encode('utf-8')).hexdigest()


def scan_docs() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    root = Config.DOCS_PATH
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if not name.lower().endswith('.docx'):
                continue
            full = os.path.join(dirpath, name)
            try:
                st = os.stat(full)
            except FileNotFoundError:
                continue
            rel = canonical_relpath(full)
            out[rel] = {
                'path': full,
                'size': st.st_size,
                'mtime': int(st.st_mtime),
                # hash computed later on demand
            }
    return out


def run_sync(manifest_path: Optional[str] = None, init_manifest: bool = False, dry_run: bool = False) -> Dict[str, Any]:
    """Programmatic sync entry.

    Returns a dict report with keys like:
      {
        'scanned': N,
        'new_or_changed': [...],
        'deleted': [...],
        'applied_upserts': int,
        'applied_deletes': int,
        'init_manifest': bool,
        'dry_run': bool,
      }
    """
    if not manifest_path:
        manifest_path = _default_manifest_path()

    report: Dict[str, Any] = {
        'scanned': 0,
        'new_or_changed': [],
        'deleted': [],
        'applied_upserts': 0,
        'applied_deletes': 0,
        'init_manifest': init_manifest,
        'dry_run': dry_run,
        'manifest': manifest_path,
    }

    manifest = load_manifest(manifest_path)
    current = scan_docs()
    report['scanned'] = len(current)

    if init_manifest:
        for rel, info in current.items():
            info['hash'] = sha256_file(info['path'])
            manifest[rel] = {
                'hash': info['hash'],
                'size': info['size'],
                'mtime': info['mtime'],
                'updated_at': int(time.time()),
            }
        save_manifest(manifest_path, manifest)
        return report

    prev_paths = set(manifest.keys())
    cur_paths = set(current.keys())
    deleted = sorted(prev_paths - cur_paths)
    candidates = sorted(cur_paths)

    new_or_changed = []
    for rel in candidates:
        cur = current[rel]
        prev = manifest.get(rel)
        need_hash = False
        if not prev:
            need_hash = True
        else:
            if prev.get('size') != cur.get('size') or prev.get('mtime') != cur.get('mtime'):
                need_hash = True
        if need_hash:
            cur_hash = sha256_file(cur['path'])
            cur['hash'] = cur_hash
            if not prev or prev.get('hash') != cur_hash:
                new_or_changed.append(rel)
        else:
            cur['hash'] = prev.get('hash')

    report['new_or_changed'] = new_or_changed
    report['deleted'] = deleted

    if dry_run:
        return report

    # Upserts
    for rel in new_or_changed:
        full = current[rel]['path']
        upsert_ird_document(full)
        manifest[rel] = {
            'hash': current[rel]['hash'],
            'size': current[rel]['size'],
            'mtime': current[rel]['mtime'],
            'updated_at': int(time.time()),
        }
        report['applied_upserts'] += 1

    # Deletions
    if deleted:
        from rag_logic import _open_db_for_update
        db = _open_db_for_update()
        for rel in deleted:
            full = os.path.join(Config.DOCS_PATH, rel)
            # Prefer manifest-recorded hash (content hash at time of last sync)
            prev_rec = manifest.get(rel) or {}
            prev_hash = prev_rec.get('hash')
            # 1) Delete by previously recorded content-hash if present
            if prev_hash:
                try:
                    db.delete(where={"attachment_group_id": prev_hash})
                except Exception:
                    pass
            # 2) Best-effort: compute current group id (falls back to path-based)
            try:
                group_id = compute_attachment_group_id(full)
                if group_id:
                    try:
                        db.delete(where={"attachment_group_id": group_id})
                    except Exception:
                        pass
            except Exception:
                pass
            # 3) Legacy ids: stem-based and path-based sha1
            try:
                import hashlib
                legacy_gid = hashlib.sha1(os.path.splitext(os.path.basename(full))[0].encode('utf-8')).hexdigest()
                db.delete(where={"attachment_group_id": legacy_gid})
            except Exception:
                pass
            try:
                rel_no_ext = os.path.splitext(rel.replace("\\", "/"))[0]
                legacy_path_gid = hashlib.sha1(rel_no_ext.encode('utf-8')).hexdigest()
                db.delete(where={"attachment_group_id": legacy_path_gid})
            except Exception:
                pass
            # Finally drop manifest entry
            manifest.pop(rel, None)
        try:
            db.persist()
        except Exception:
            pass
        report['applied_deletes'] = len(deleted)

    save_manifest(manifest_path, manifest)
    return report


def _collect_vectorstore_group_ids() -> Dict[str, Any]:
    """Return a mapping with present attachment_group_ids and a sample label per id.

    { 'groups': set([...]), 'labels': {group_id: 'parent_docx_basename or filename'} }
    """
    client = PersistentClient(path=Config.DB_NAME)
    db = Chroma(
        client=client,
        collection_name=Config.COLLECTION_NAME,
        embedding_function=get_embedding_model(),
    )
    col = db._collection
    groups = set()
    labels: Dict[str, str] = {}
    try:
        total = col.count()
        offset = 0
        page = 1000
        while offset < total:
            res = col.get(include=["metadatas"], limit=page, offset=offset)
            metas = res.get("metadatas", []) or []
            ids = res.get("ids", []) or []
            for m in metas:
                gid = (m or {}).get("attachment_group_id")
                if gid:
                    groups.add(gid)
                    if gid not in labels:
                        labels[gid] = m.get("parent_docx_basename") or m.get("filename") or ""
            offset += len(ids) if ids else len(metas)
    except Exception:
        # Fallback: fetch by ids in chunks
        res = col.get(include=[])
        ids = res.get("ids", []) if isinstance(res, dict) else []
        for i in range(0, len(ids), 1000):
            sl = ids[i:i+1000]
            res = col.get(ids=sl, include=["metadatas"])
            for m in res.get("metadatas", []) or []:
                gid = (m or {}).get("attachment_group_id")
                if gid:
                    groups.add(gid)
                    if gid not in labels:
                        labels[gid] = m.get("parent_docx_basename") or m.get("filename") or ""
    return {"groups": groups, "labels": labels}


def _collect_db_docs_by_relpath() -> Dict[str, Dict[str, Any]]:
    """Collect unique docs from the vector store keyed by docx_relpath.

    Returns mapping:
      { relpath: { 'hash': sha256, 'any_label': str } }
    Only includes entries that have docx_relpath in metadata.
    """
    client = PersistentClient(path=Config.DB_NAME)
    db = Chroma(
        client=client,
        collection_name=Config.COLLECTION_NAME,
        embedding_function=get_embedding_model(),
    )
    col = db._collection
    out: Dict[str, Dict[str, Any]] = {}
    try:
        total = col.count()
        offset = 0
        page = 1000
        while offset < total:
            res = col.get(include=["metadatas"], limit=page, offset=offset)
            metas = res.get("metadatas", []) or []
            ids = res.get("ids", []) or []
            for m in metas:
                rel = (m or {}).get('docx_relpath')
                if not rel:
                    continue
                h = (m or {}).get('doc_content_sha256') or (m or {}).get('attachment_group_id')
                if rel not in out:
                    out[rel] = {
                        'hash': h,
                        'any_label': (m or {}).get('parent_docx_basename') or (m or {}).get('filename') or '',
                    }
            offset += len(ids) if ids else len(metas)
    except Exception:
        # Fallback by ids paging
        res = col.get(include=[])
        ids = res.get("ids", []) if isinstance(res, dict) else []
        for i in range(0, len(ids), 1000):
            sl = ids[i:i+1000]
            res = col.get(ids=sl, include=["metadatas"])
            for m in res.get("metadatas", []) or []:
                rel = (m or {}).get('docx_relpath')
                if not rel:
                    continue
                h = (m or {}).get('doc_content_sha256') or (m or {}).get('attachment_group_id')
                if rel not in out:
                    out[rel] = {
                        'hash': h,
                        'any_label': (m or {}).get('parent_docx_basename') or (m or {}).get('filename') or '',
                    }
    return out


def run_init_from_db(manifest_path: Optional[str] = None) -> Dict[str, Any]:
    """Initialize the manifest from documents already present in the vector store.

    Uses docx_relpath and doc_content_sha256 from chunk metadata (new pipeline),
    and falls back to attachment_group_id for hash.
    """
    if not manifest_path:
        manifest_path = _default_manifest_path()
    docs = _collect_db_docs_by_relpath()
    root = Config.DOCS_PATH
    manifest: Dict[str, Any] = {}
    for rel, info in docs.items():
        full = os.path.join(root, rel)
        try:
            st = os.stat(full)
            size = st.st_size
            mtime = int(st.st_mtime)
        except Exception:
            # If file is missing on disk, still write with size/mtime=None
            size = None
            mtime = None
        manifest[rel] = {
            'hash': info.get('hash'),
            'size': size,
            'mtime': mtime,
            'updated_at': int(time.time()),
        }
    save_manifest(manifest_path, manifest)
    return {
        'ok': True,
        'initialized_from_db': True,
        'written': len(manifest),
        'manifest': manifest_path,
    }

def run_verify() -> Dict[str, Any]:
    """Compare expected docs (from filesystem) with vector store content.

    Returns a report with missing (in vector store) and stale (in vector store but not on disk).
    """
    current = scan_docs()
    expected_groups = {}
    for rel, info in current.items():
        gid = compute_attachment_group_id(os.path.join(Config.DOCS_PATH, rel))
        expected_groups[gid] = rel

    present = _collect_vectorstore_group_ids()
    present_groups = present["groups"]
    labels = present["labels"]

    missing = []
    for gid, rel in expected_groups.items():
        if gid not in present_groups:
            missing.append(rel)

    stale = []
    for gid in present_groups:
        if gid not in expected_groups:
            stale.append({"group_id": gid, "label": labels.get(gid, "")})

    return {
        "total_files": len(current),
        "present_groups": len(present_groups),
        "missing_count": len(missing),
        "stale_count": len(stale),
        "missing": sorted(missing),
        "stale": stale,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=_default_manifest_path(), help='Path to JSON manifest')
    ap.add_argument('--dry-run', action='store_true', help='Only report changes; do not write or ingest')
    ap.add_argument('--init-manifest', action='store_true', help='Write a baseline manifest from current files and exit (no upserts/deletes)')
    ap.add_argument('--verify', action='store_true', help='Verify vector store contains all docs (no changes applied)')
    ap.add_argument('--init-from-db', action='store_true', help='Initialize manifest from existing vector store metadata')
    args = ap.parse_args()

    if args.init_from_db:
        rep = run_init_from_db(args.manifest)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0
    if args.verify:
        v = run_verify()
        print(json.dumps(v, ensure_ascii=False, indent=2))
        return 0
    report = run_sync(args.manifest, init_manifest=args.init_manifest, dry_run=args.dry_run)
    if args.init_manifest:
        print(f"Initialized manifest with {report.get('scanned', 0)} document(s) at {args.manifest}")
        return 0

    # Build sets
    print(f"Found {report.get('scanned', 0)} docx, new/changed={len(report.get('new_or_changed', []))}, deleted={len(report.get('deleted', []))}")

    if args.dry_run:
        if report.get('new_or_changed'):
            print("Would upsert:")
            for rel in report['new_or_changed']:
                print("  +", rel)
        if report.get('deleted'):
            print("Would delete from vector store:")
            for rel in report['deleted']:
                print("  -", rel)
        return 0

    # Process upserts
    for rel in report.get('new_or_changed', []):
        print(f"Upserted: {rel}")

    # Process deletions: remove vectors by attachment_group_id and drop manifest record
    if report.get('deleted'):
        for rel in report['deleted']:
            print(f"Deleted: {rel}")
    print("Sync complete.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
