import os
import json
import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

from typing import Any, Dict

import redis

from docx import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

from config import Config
from rag_logic import get_llm, get_embedding_model
from read_split_nbu import split_docx_to_question_with_ids
from save_results import add_comment_to_paragraphs, sanitize_markdown
from rag_functions import build_packed_context, source_label_from_meta
from cache_manifest import cache_store, sha256_file


def _color_from_answer(ans_emoji: str) -> str:
    if ans_emoji == "✅":
        return "green"
    if ans_emoji == "❌":
        return "red"
    return "yellow"



def _setup_worker_logging():
    """Attach a rotating file handler for worker logs (same file as app)."""
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    fmt = '{"t":"%(asctime)s","level":"%(levelname)s","msg":%(message)r}'
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if str(os.getenv("DEBUG", "0")).lower() in ("1","true","yes") else logging.INFO)
    want_path = os.path.abspath(os.path.join(log_dir, "app.log"))
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, 'baseFilename', None) == want_path for h in root.handlers):
        fh = RotatingFileHandler(want_path, maxBytes=10*1024*1024, backupCount=5)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)


_setup_worker_logging()


def _task_key(task_id: str) -> str:
    return f"task:{task_id}"


def _task_events_key(task_id: str) -> str:
    return f"task_events:{task_id}"


def _redis():
    return redis.from_url(Config.REDIS_URL, decode_responses=True)


def _make_subqueries(q: str) -> list[str]:
    from rag_functions import make_subqueries
    return make_subqueries(q)


def _checkpoint_path(task_id: str) -> str:
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(Config.CHECKPOINT_DIR, f"{task_id}.json")


def _load_checkpoint(task_id: str) -> Dict[str, Any] | None:
    p = _checkpoint_path(task_id)
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_checkpoint(task_id: str, data: Dict[str, Any]) -> None:
    p = _checkpoint_path(task_id)
    data = {**data, "updated_at": datetime.utcnow().isoformat() + "Z"}
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def _delete_checkpoint(task_id: str) -> None:
    p = _checkpoint_path(task_id)
    try:
        os.remove(p)
    except Exception:
        pass


def rq_process_document_task(task_id: str):
    """RQ worker entrypoint. Resumable via JSON checkpointing.

    Reads task metadata from Redis, processes buckets sequentially, persists progress
    to Config.CHECKPOINT_DIR/<task_id>.json, and pushes SSE events to Redis list.
    """
    logging.info(f"RQ start task_id=%s", task_id)
    r = _redis()

    t = r.hgetall(_task_key(task_id))
    if not t:
        logging.error("Task metadata not found in Redis for task_id=%s", task_id)
        return

    filename_in = t.get("filename_in")
    filename_out = t.get("filename_out")
    if not filename_in or not os.path.exists(filename_in):
        r.hset(_task_key(task_id), mapping={"status": "error", "finished_at": str(time.time())})
        r.rpush(_task_events_key(task_id), json.dumps({"status": "error", "error": "input_missing"}))
        r.ltrim(_task_events_key(task_id), -200, -1)
        return

    # Open DB + LLM for worker
    DB = Chroma(
        collection_name=Config.COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=Config.DB_NAME
    )
    LLM_DOC = get_llm(kind='document')

    # Prepare buckets (questions with para indices)
    buckets = split_docx_to_question_with_ids(filename_in, second_split=True)
    total = len(buckets)

    # Load or init checkpoint
    ck = _load_checkpoint(task_id) or {
        "task_id": task_id,
        "filename_in": filename_in,
        "filename_out": filename_out,
        "total": total,
        "done": 0,
        "results": [],
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    # Reconcile total (if splitter changed)
    ck["total"] = total
    done = int(ck.get("done") or 0)
    results = list(ck.get("results") or [])

    # Start or resume DOCX
    doc = Document(filename_in)
    # Reapply previous comments so the final output accumulates
    for res in results:
        try:
            color = _color_from_answer(res.get("verdict", "❓"))
            para_indices0 = res.get("para_indices0") or []
            text_answer = res.get("text_answer") or ""
            sources = res.get("sources") or []
            comment_text = (
                f"{res.get('verdict','❓')} {text_answer.strip()}\n\nДжерела:\n" +
                ("\n".join(f"- {s}" for s in sources) if isinstance(sources, list) else str(sources))
            )
            if para_indices0:
                add_comment_to_paragraphs(doc, para_indices0, comment_text, author="RAG Assistant", initials="AI", color=color)
        except Exception:
            pass

    # Shared system prompt
    system_prompt = Config.system_prompt_document_loop

    # Update Redis state
    r.hset(_task_key(task_id), mapping={"status": "running", "total": total, "done": done, "started_at": str(time.time())})
    r.rpush(_task_events_key(task_id), json.dumps({"status": "running"}))
    r.ltrim(_task_events_key(task_id), -200, -1)

    # Continue from last done index
    for idx in range(done, total):
        b = buckets[idx]
        q_text = b.get("question_text", "")
        para_indices0 = b.get("para_indices0", []) or b.get("para_ids", [])
        if not isinstance(para_indices0, list):
            para_indices0 = []

        packed, _ = build_packed_context(DB, q_text)

        context_str = "\n\n".join(
            f"===\n[Джерело: {source_label_from_meta(d.metadata)}]\n{d.page_content}\n===" for d in packed
        )

        prompt = f"""{system_prompt}
Контекст з ВНД:
{context_str}

Вимога НБУ:
{q_text}

Відповідь українською:
"""

        # LLM call
        # LLM call with simple retry/backoff
        attempt = 0
        delay = 2.0
        while True:
            try:
                resp = LLM_DOC.invoke(prompt)
                break
            except Exception as e:
                attempt += 1
                if attempt >= 5:
                    logging.exception("LLM invoke failed after retries: %s", e)
                    raise
                # push retry hint to progress stream
                try:
                    r.rpush(_task_events_key(task_id), json.dumps({
                        "task_id": task_id,
                        "status": "running",
                        "retry": attempt,
                        "retry_in": int(delay),
                        "phase": "llm_retry"
                    }))
                    r.ltrim(_task_events_key(task_id), -200, -1)
                except Exception:
                    pass
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raw = resp.content if hasattr(resp, "content") else str(resp)

        verdict = "❓"; text_answer = raw; sources = []
        try:
            data = json.loads(raw)
            text_answer = sanitize_markdown(data.get("text", raw))
            sources = data.get("source", [])
            verdict = data.get("answer", "❓")
            if verdict not in ("✅", "❌", "❓"):
                verdict = "❓"
            verdict = (verdict or "").replace("\x00", "")
        except Exception:
            pass

        color = _color_from_answer(verdict)

        # Add comment
        if para_indices0:
            try:
                comment_text = (
                    f"{verdict} {text_answer.strip()}\n\nДжерела:\n" +
                    ("\n".join(f"- {s}" for s in sources) if isinstance(sources, list) else str(sources))
                )
                add_comment_to_paragraphs(doc, para_indices0, comment_text, author="RAG Assistant", initials="AI", color=color)
            except Exception:
                pass

        # Update checkpoint
        results.append({
            "index": idx,
            "question_text": q_text,
            "para_indices0": para_indices0,
            "verdict": verdict,
            "text_answer": text_answer,
            "sources": sources,
        })
        done = idx + 1
        ck.update({"done": done, "results": results})
        _save_checkpoint(task_id, ck)

        # Push progress event
        r.hset(_task_key(task_id), mapping={"done": str(done)})
        r.rpush(_task_events_key(task_id), json.dumps({
            "task_id": task_id,
            "status": "running",
            "total": total,
            "done": done,
            "last_verdict": verdict,
            "last_brief": (text_answer[:140] + "…") if len(text_answer) > 140 else text_answer
        }))
        r.ltrim(_task_events_key(task_id), -200, -1)

    # Save DOCX output
    os.makedirs(os.path.dirname(filename_out), exist_ok=True)
    doc.save(filename_out)

    # Finalize
    r.hset(_task_key(task_id), mapping={"status": "done", "finished_at": str(time.time())})
    r.rpush(_task_events_key(task_id), json.dumps({"status": "done", "task_id": task_id}))
    r.ltrim(_task_events_key(task_id), -200, -1)
    _delete_checkpoint(task_id)
    # Persist processed-file cache mapping by input content hash
    try:
        ihash = sha256_file(filename_in)
        cache_store(ihash, filename_out, meta={
            "task_id": task_id,
        })
    except Exception:
        pass
    logging.info("RQ done task_id=%s", task_id)
