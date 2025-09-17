import sys
import time
import tiktoken
import itertools
import re

from typing import List, Iterable, Tuple
from operator import itemgetter

from langchain.schema import Document as LangchainDocument

from langchain_community.retrievers import BM25Retriever
from collections import defaultdict, deque

from colorama import Fore  # Terminal colors

from config import Config

token_usage_log = deque()

# Local debug helper (opt-in via Config.DEBUG)
DEBUG = getattr(Config, "DEBUG", False)

def dprint(msg: str):
    if DEBUG:
        print(msg)


def rerank_by_keyword_overlap(query: str, docs: List[LangchainDocument]) -> List[LangchainDocument]:
    """Very simple keyword-overlap re-ranker.

    Splits query/docs into lowercased whitespace tokens and ranks by intersection size.
    """
    dprint("rerank_by_keyword_overlap()")
    query_words = set(query.lower().split())
    ranked: List[Tuple[int, LangchainDocument]] = []
    for doc in docs:
        doc_words = set(doc.page_content.lower().split())
        score = len(query_words & doc_words)
        ranked.append((score, doc))
    ranked.sort(reverse=True, key=itemgetter(0))
    return [doc for score, doc in ranked]





def rrf_fuse(lists: Iterable[Iterable[LangchainDocument]], k: int = 60, c: float = 60.0) -> List[LangchainDocument]:
    """Reciprocal Rank Fusion over lists of LangchainDocument.

    Args:
        lists: An iterable of ranked lists of documents.
        k: Max number of documents to return.
        c: RRF constant; higher reduces impact of rank position.
    """
    dprint("rrf_fuse()")
    scores: defaultdict = defaultdict(float)
    order: List[Tuple[str, int, str]] = []
    for lst in lists:
        for rank, doc in enumerate(lst, 1):
            doc_id = (doc.metadata.get("filename"), doc.metadata.get("chunk_index"), doc.page_content[:64])
            if doc_id not in order:
                order.append(doc_id)
            scores[doc_id] += 1.0 / (c + rank)
    merged: List[LangchainDocument] = []
    for doc_id in sorted(order, key=lambda d: -scores[d]):
        # return the first actual doc object that matches the id
        for lst in lists:
            for d in lst:
                if (d.metadata.get("filename"), d.metadata.get("chunk_index"), d.page_content[:64]) == doc_id:
                    merged.append(d)
                    break
            else:
                continue
            break
        if len(merged) >= k:
            break
    return merged

def build_bm25_retriever(all_docs: List[LangchainDocument]) -> BM25Retriever:
    """In‑memory BM25 retriever over ALL docs (LangchainDocument)."""
    dprint("build_bm25_retriever()")
    retriever = BM25Retriever.from_documents(all_docs)
    retriever.k = Config.HYBRID_BM25_K
    return retriever

def cap_per_ird(docs: Iterable[LangchainDocument], max_per_doc: int = 5) -> List[LangchainDocument]:
    """Limit number of chunks per document source to avoid dominance."""
    dprint("cap_per_ird()")
    out: List[LangchainDocument] = []
    seen: dict = {}
    for d in docs:
        fn = d.metadata.get("filename") or d.metadata.get("source") or "unknown"
        seen[fn] = seen.get(fn, 0) + 1
        if seen[fn] <= max_per_doc:
            out.append(d)
    return out

def pack_context(docs: Iterable[LangchainDocument], max_prompt_tokens: int, avg_tok: int = 450) -> Tuple[List[LangchainDocument], int]:
    """Greedy pack on estimated tokens.

    Prefer chunk metadata fields produced by our pipeline:
      - token_count_o200k (from o200k tokenizer)
      - token_count_cl100k
      - token_count (fallback, if present)
    Else, fall back to avg_tok.
    """
    dprint("pack_context()")
    out: List[LangchainDocument] = []
    used = 0
    for d in docs:
        t = (
            d.metadata.get("token_count_o200k")
            or d.metadata.get("token_count_cl100k")
            or d.metadata.get("token_count")
            or avg_tok
        )
        if used + t > max_prompt_tokens:
            break
        out.append(d)
        used += t
    return out, used

# ---------- Consolidated helpers ----------

def make_subqueries(q: str) -> list[str]:
    """Split a query into line items, also include full query; deduplicate preserving order."""
    dprint("make_subqueries()")
    parts = [x.strip() for x in q.splitlines() if x.strip()]
    out = []
    for line in parts:
        if re.match(r'^\s*(?:\d+(\.\d+)*\.|\-|\•)\s+', line):
            out.append(re.sub(r'^\s*(?:\d+(\.\d+)*\.|\-|\•)\s+', '', line))
        else:
            out.append(line)
    if q not in out:
        out.append(q)
    seen, dedup = set(), []
    for s in out:
        if s not in seen:
            dedup.append(s)
            seen.add(s)
    return dedup

def source_label_from_meta(meta: dict, mode: str | None = None) -> str:
    """Return source label based on mode: 'filename' (default) or 'docx_relpath'."""
    try:
        mode_eff = (mode or getattr(Config, 'SOURCE_LABEL_MODE', 'filename')).strip().lower()
        if mode_eff == 'docx_relpath':
            v = (meta or {}).get('docx_relpath')
            if v:
                return v
        return (meta or {}).get('filename') or 'невідомо'
    except Exception:
        return (meta or {}).get('filename') or 'невідомо'

def build_packed_context(db, query_text: str,
                         reserve_tokens: int = 1200,
                         max_cap: int = 18000,
                         min_floor: int = 2000,
                         avg_tok: int = 450) -> Tuple[List[LangchainDocument], int]:
    """Build a hybrid (dense+sparse) context and pack within token budget.

    Returns (packed_docs, used_tokens_estimate).
    """
    dprint("build_packed_context()")
    # Build dense retriever per call to allow dynamic params
    dense = db.as_retriever(
        search_kwargs={
            "k": getattr(Config, "HYBRID_DENSE_K", 12),
            "fetch_k": getattr(Config, "HYBRID_DENSE_K", 12) * 3,
        },
        search_type="mmr",
    )
    subqueries = make_subqueries(query_text)
    dense_lists = [dense.invoke(sq) for sq in subqueries]

    # Build BM25 on union of dense candidates to stay memory-light
    seen = set(); candidate_pool: List[LangchainDocument] = []
    for lst in dense_lists:
        for d in lst:
            did = (d.metadata.get("filename"), d.metadata.get("chunk_index"), d.page_content[:64])
            if did in seen:
                continue
            seen.add(did)
            candidate_pool.append(d)

    bm25_local = BM25Retriever.from_documents(candidate_pool)
    bm25_local.k = getattr(Config, "HYBRID_BM25_K", 200)
    sparse_lists = [bm25_local.get_relevant_documents(sq) for sq in subqueries]

    fused = rrf_fuse(dense_lists + sparse_lists, k=getattr(Config, "RRF_MERGE_K", 60))
    fused_capped = cap_per_ird(fused, max_per_doc=getattr(Config, "PER_IRD_CAP", 5))

    max_ctx_tokens = max(min_floor, min(getattr(Config, 'MAX_TOKENS_PER_REQUEST', 22000) - reserve_tokens, max_cap))
    packed, used = pack_context(fused_capped, max_ctx_tokens, avg_tok=avg_tok)
    return packed, used

# def hybrid_retrieve(db, all_docs, subqueries):
#     print(f"{'='*33} def hybrid_retrieve(db, all_docs, subqueries): {'='*33}")


#     """
#     For each subquery: dense K + bm25 K -> RRF -> collect.
#     Flatten, then cap per IRD. Return the capped list.
#     """
#     dense_ret = db.as_retriever(search_kwargs={"k": Config.HYBRID_DENSE_K})
#     bm25 = build_bm25_retriever(all_docs)

#     fused_lists = []
#     for q in subqueries:
#         dense = dense_ret.invoke(q)
#         sparse = bm25.get_relevant_documents(q)
#         fused = rrf_fuse([dense, sparse], k=Config.RRF_MERGE_K)
#         fused_lists.append(fused)

#     flat = [d for lst in fused_lists for d in lst]
#     return cap_per_ird(flat, max_per_doc=Config.PER_IRD_CAP)






def count_tokens(text: str, model_name: str = "gpt-4o") -> int:
    """Return token count using tiktoken, with safe fallback."""
    dprint("count_tokens()")
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        # Fallback to a common encoding if the specific model name is unknown
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def spinner(msg: str, stop_event):
    """Console spinner while waiting for an operation."""
    dprint("spinner()")
    spinner_cycle = itertools.cycle(['|', '/', '-', '\\'])
    while not stop_event.is_set():
        sys.stdout.write(f"\r{msg} {next(spinner_cycle)}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * (len(msg) + 2) + "\r")

def wait_if_tpm_exceeded(estimated_tokens: int) -> None:
    """Best-effort rate limiting guard based on estimated tokens/minute.

    Uses a simple deque of (timestamp, tokens) entries. Caller is responsible
    for appending actual usage when available.
    """
    dprint("wait_if_tpm_exceeded()")
    now = time.time()
    # Remove outdated entries
    while token_usage_log and now - token_usage_log[0][0] > Config.TOKEN_WINDOW_SECONDS:
        token_usage_log.popleft()

    tokens_last_minute = sum(tokens for t, tokens in token_usage_log)

    if tokens_last_minute + estimated_tokens > Config.MAX_TOKENS_PER_MINUTE and token_usage_log:
        wait_time = Config.TOKEN_WINDOW_SECONDS - (now - token_usage_log[0][0])
        print(Fore.RED + f"⚠️ TPM limit would be exceeded. Sleeping {round(wait_time, 2)}s..." + Fore.RESET)
        time.sleep(wait_time)
        # After waiting, clean up old entries again
        now = time.time()
        while token_usage_log and now - token_usage_log[0][0] > Config.TOKEN_WINDOW_SECONDS:
            token_usage_log.popleft()

def question_giver(questions, start_from_question: int = Config.START_QUESTION, question_skip_list = Config.QUESTION_SKIP_LIST):
    """Yield questions with support for start index and skip list."""
    dprint("question_giver()")
    print(f"start_from_question : {start_from_question}, question_skip_list : {question_skip_list}")
    questions_len = len(questions)
    if start_from_question > questions_len:
        raise IndexError("Start question is out of boundaries")
    for i, question in enumerate(questions, 1):
        if question_skip_list and i in question_skip_list:
            continue
        if i < start_from_question:
            continue
        print(Fore.YELLOW + f"question: {i}/{questions_len}" + Fore.RESET)
        yield question



# def question_giver_s(questions, start_from_question=Config.START_QUESTION, stop_on_question=Config.STOP_QUESTION, question_skip_list = Config.QUESTION_SKIP_LIST):
#     print(f"{'='*33} def question_giver_s(questions, start_from_question=Config.START_QUESTION, stop_on_question=Config.STOP_QUESTION, question_skip_list = Config.QUESTION_SKIP_LIST): {'='*33}")

    
#     print(f"start_from_question : {start_from_question}, stop_on_question : {stop_on_question}, question_skip_list : {question_skip_list}")
#     questions_len = len(questions)
#     if stop_on_question == 0:
#         # questions_len = len(questions)
#         stop_on_question = len(questions)
#     elif start_from_question > stop_on_question:
#         raise IndexError("Start question is out of boundaries")
#         return
#     else:
#         questions_len = stop_on_question - start_from_question
#     if start_from_question > questions_len:
#         raise IndexError("Start question is out of boundaries")
#         return
#     for i, question in enumerate(questions, 1):
#         if question_skip_list:
#             if i in question_skip_list:
#                 continue
#         if i < start_from_question:
#             continue
#         if i > stop_on_question:
#             continue
#         print(Fore.YELLOW + f"question: {i}/{questions_len}" + Fore.RESET)
#         yield question
