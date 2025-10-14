from typing import List, Dict, Optional

# from chunking_evaluation.chunking import RecursiveTokenChunker
# from chunking_evaluation.utils import openai_token_count

from recursive_token_chunker import RecursiveTokenChunker, openai_token_count

import os
import re
import json
import uuid
import zipfile
import tiktoken
import pypandoc
from pprint import pprint

from colorama import Fore
from langchain.schema import Document as LangchainDocument

from config import Config

from docx_to_md import run_logic as convert_docx_to_markdown


CHUNK1 = 10
CHUNK2 = 11

CHUNK_SIZE = Config.CHUNK_SIZE
CHUNK_OVERLAP = Config.CHUNK_OVERLAP

# Primary tokenizer/encoding used for chunk budgeting
ENCODING_MODEL = "o200k_base"

# Adjusted chunking controls (token-level)
TARGET_CHUNK_TOKENS = max(256, CHUNK_SIZE)
MIN_CHUNK_TOKENS = max(120, TARGET_CHUNK_TOKENS // 3)
HARD_CHUNK_TOKENS = max(TARGET_CHUNK_TOKENS + 128, int(TARGET_CHUNK_TOKENS * 1.35))
OVERLAP_TOKENS = max(0, min(CHUNK_OVERLAP, TARGET_CHUNK_TOKENS // 2))

# Debug gating via Config.DEBUG
DEBUG = getattr(Config, "DEBUG", False)

def dprint(msg: str):
    if DEBUG:
        print(msg)


_TOKEN_ENCODER = tiktoken.get_encoding(ENCODING_MODEL)
_CL100K_ENCODER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKEN_ENCODER.encode(text, disallowed_special=()))


def _tail_tokens(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    tokens = _TOKEN_ENCODER.encode(text, disallowed_special=())
    if not tokens:
        return ""
    if len(tokens) <= budget:
        return text
    tail = tokens[-budget:]
    return _TOKEN_ENCODER.decode(tail)


def _sha1(text: str) -> str:
    dprint(f"{'='*33} def _sha1(text) {'='*33}")
    import hashlib
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def _sha256_file(path: str) -> Optional[str]:
    try:
        import hashlib
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

_heading_re = re.compile(r'^\s{0,3}(#{1,6})\s+(.*)$', re.MULTILINE)


_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


def _classify_markdown_line(line: str, in_code_block: bool) -> str:
    stripped = line.strip()
    if in_code_block:
        return "code"
    if not stripped:
        return "blank"
    if stripped.startswith("```"):
        return "fence"
    if stripped.startswith("#"):
        return "heading"
    if _TABLE_LINE_RE.match(stripped):
        return "table"
    if _LIST_LINE_RE.match(stripped):
        return "list"
    if stripped.startswith(">"):
        return "quote"
    return "paragraph"


def _split_markdown_into_blocks(text: str) -> List[str]:
    """Coalesce markdown lines into coherent blocks (tables, lists, paragraphs)."""
    lines = text.splitlines()
    blocks: List[str] = []
    current: List[str] = []
    current_type: Optional[str] = None
    in_code = False

    def flush() -> None:
        nonlocal current, current_type
        if current:
            block = "\n".join(current).strip("\n")
            if block:
                blocks.append(block)
        current = []
        current_type = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            line_type = _classify_markdown_line(line, in_code)
            if in_code:
                current.append(line)
                flush()
                in_code = False
            else:
                flush()
                current.append(line)
                current_type = "code"
                in_code = True
            continue

        line_type = _classify_markdown_line(line, in_code)
        if line_type == "blank" and not in_code:
            flush()
            continue
        if current_type is None:
            current_type = line_type
            current.append(line)
            continue
        if line_type != current_type and not in_code:
            flush()
            if line_type == "blank":
                continue
            current_type = line_type
            current.append(line)
        else:
            current.append(line)
    flush()
    return blocks


def _fallback_split(text: str) -> List[str]:
    """Fallback to the recursive chunker for very large blocks."""
    chunker = RecursiveTokenChunker(
        chunk_size=min(HARD_CHUNK_TOKENS, TARGET_CHUNK_TOKENS * 2),
        chunk_overlap=min(OVERLAP_TOKENS, 64),
        length_function=openai_token_count,
        separators=["\n\n", "\n", " ", ""],
    )
    parts = [p.strip() for p in chunker.split_text(text) if p.strip()]
    return parts or [text]

def _stable_find_positions(big: str, pieces: List[str]) -> List[tuple]:
    dprint(f"{'='*33} def _stable_find_positions(...) {'='*33}")

    """
    Find non-overlapping positions of each chunk in order.
    Prefers matches at/after the previous end; falls back to global search.
    Returns list of (start, end); (-1, -1) if not found.
    """
    positions = []
    cursor = 0
    backtrack = 2000
    for chunk in pieces:
        idx = big.find(chunk, max(cursor - backtrack, 0))
        if idx == -1:
            idx = big.find(chunk)
        if idx == -1:
            positions.append((-1, -1))
        else:
            positions.append((idx, idx + len(chunk)))
            cursor = idx + len(chunk)
    return positions

def _extract_heading_path_for_chunks(chunks: List[str]) -> List[Dict]:
    dprint(f"{'='*33} def _extract_heading_path_for_chunks(...) {'='*33}")

    """
    For each chunk, compute:
    - headings_in_chunk: all md headings found in the chunk (top->bottom).
    - heading_path: inherited H1..H6 path snapshot active at this chunk.
    """
    path = [""] * 6  # last seen titles for levels 1..6
    out: List[Dict] = []
    for chunk in chunks:
        found = []
        for m in _heading_re.finditer(chunk):
            level = len(m.group(1))
            title = m.group(2).strip()
            found.append(title)
            path[level - 1] = title
            # clear deeper levels
            for i in range(level, 6):
                if i != (level - 1):
                    path[i] = ""
        effective_path = [p for p in path if p]
        out.append({"headings_in_chunk": found, "heading_path": effective_path})
    return out


def _merge_small_chunks(chunks: List[str]) -> List[str]:
    if not chunks:
        return chunks
    merged: List[str] = []
    for chunk in chunks:
        tokens = _count_tokens(chunk)
        if merged and tokens < MIN_CHUNK_TOKENS:
            candidate = merged[-1] + "\n\n" + chunk
            if _count_tokens(candidate) <= HARD_CHUNK_TOKENS:
                merged[-1] = candidate
                continue
        merged.append(chunk)
    return merged


def _build_chunks(blocks: List[str]) -> List[str]:
    results: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def finalize(carry_overlap: bool) -> None:
        nonlocal current, current_tokens
        if not current:
            return
        chunk_text = "\n\n".join(current).strip()
        if not chunk_text:
            current = []
            current_tokens = 0
            return
        results.append(chunk_text)
        if carry_overlap and OVERLAP_TOKENS > 0:
            overlap_text = _tail_tokens(chunk_text, OVERLAP_TOKENS)
            if overlap_text:
                current = [overlap_text]
                current_tokens = _count_tokens(overlap_text)
                return
        current = []
        current_tokens = 0

    idx = 0
    while idx < len(blocks):
        block = blocks[idx]
        if not block.strip():
            idx += 1
            continue
        block_tokens = _count_tokens(block)
        if block_tokens > HARD_CHUNK_TOKENS:
            sub_parts = _fallback_split(block)
            blocks[idx:idx+1] = sub_parts
            continue

        if current_tokens and current_tokens + block_tokens > HARD_CHUNK_TOKENS and current_tokens >= MIN_CHUNK_TOKENS:
            finalize(carry_overlap=True)
            continue

        if current_tokens == 0 and block_tokens >= HARD_CHUNK_TOKENS:
            results.append(block)
            idx += 1
            continue

        current.append(block)
        current_tokens += block_tokens

        if current_tokens >= TARGET_CHUNK_TOKENS:
            finalize(carry_overlap=True)
        idx += 1

    finalize(carry_overlap=False)

    return _merge_small_chunks(results)

# --- Attachment/parent linkage helper ---
def _infer_attachment_metadata_from_md_path(file_path: str) -> Dict:
    dprint(f"{'='*33} def _infer_attachment_metadata_from_md_path(file_path) {'='*33}")

    """Infer linkage and department metadata from MD file path.

    Path-safe grouping:
    - attachment_group_id = sha1(relative_docx_path_without_extension)
      where relative_docx_path is under Config.DOCS_PATH.
    - Adds md_relpath (relative to Config.DOCS_PATH_MD) and docx_relpath (relative to Config.DOCS_PATH).
    - Recognizes `<parent>_dodatok_<i>.md` attachments and groups them to the same parent docx.
    """
    base = os.path.basename(file_path)
    if not base.lower().endswith(".md"):
        return {}
    stem = base[:-3]

    # Compute department and relative md path
    department = None
    md_relpath = None
    try:
        md_root = Config.DOCS_PATH_MD
        rel = os.path.relpath(file_path, md_root)
        md_relpath = rel
        if not rel.startswith(".."):
            parts = rel.split(os.sep)
            if len(parts) > 1:
                department = parts[0]
    except Exception:
        pass

    # Map MD relative path to DOCX relative path (mirrored structure)
    # For main md `<dir>/<name>.md` -> `<dir>/<name>.docx`
    # For attachments `<name>_dodatok_i.md` -> parent `<name>.docx`
    dir_rel = os.path.dirname(md_relpath) if md_relpath not in (None, "", ".") else ""
    is_attachment = False
    parent_stem = stem
    if "_dodatok_" in stem:
        parent_stem = stem.split("_dodatok_", 1)[0]
        is_attachment = True

    # Build docx_relpath relative to DOCS_PATH (mirror of md_relpath directory + parent stem)
    if dir_rel in (None, "", "."):
        docx_relpath = f"{parent_stem}.docx"
        parent_md_basename = f"{parent_stem}.md"
    else:
        docx_relpath = os.path.join(dir_rel, f"{parent_stem}.docx")
        parent_md_basename = os.path.join(dir_rel, f"{parent_stem}.md")

    # Compute content hash of the parent DOCX to use as stable, path-agnostic group id
    docx_abs = os.path.join(Config.DOCS_PATH, docx_relpath)
    content_hash = _sha256_file(docx_abs)
    if not content_hash:
        # Fallback to path-based hash if file not found (should be rare)
        norm_docx_rel_no_ext = os.path.splitext(docx_relpath.replace("\\", "/"))[0]
        content_hash = _sha1(norm_docx_rel_no_ext)
    group_id = content_hash

    out = {
        "attachment_group_id": group_id,
        "parent_basename": f"{parent_stem}.md",
        "parent_docx_basename": f"{parent_stem}.docx",
        "docx_relpath": docx_relpath.replace("\\", "/"),
        "doc_content_sha256": content_hash,
        "attachment_index": None,
    }
    if is_attachment:
        # Try to extract an index integer
        try:
            idx_str = stem.split("_dodatok_", 1)[1]
            out["attachment_index"] = int(idx_str)
        except Exception:
            out["attachment_index"] = None
        out["attachment_of"] = docx_relpath.replace("\\", "/")
    else:
        out["attachment_of"] = None

    if department is not None:
        out["department"] = department
    if md_relpath is not None:
        out["md_relpath"] = md_relpath.replace("\\", "/")
    return out

    if "_dodatok_" in stem:
        parent, idx = stem.rsplit("_dodatok_", 1)
        try:
            idx_int = int(idx)
        except Exception:
            idx_int = None
        return {
            "source_type": "embedded",
            "attachment_of": f"{parent}.md",
            **_common_fields(parent, idx_int),
        }
    else:
        # main md exported from `<parent>.docx` as `<parent>.md`
        parent = stem
        return {
            "source_type": "main",
            "attachment_of": None,
            **_common_fields(parent, None),
        }



def count_tokens2(text):
    dprint(f"{'='*33} def count_tokens2(text) {'='*33}")

    """Count tokens in a text string using tiktoken (console demo)."""
    model35="cl100k_base"
    encoder35 = tiktoken.get_encoding(model35)
    model4o="o200k_base"
    encoder4o = tiktoken.get_encoding(model4o)

    message = f"""
    Number of tokens (GPT-3.5 tokenizer): {len(encoder35.encode(text))}
    Number of tokens (GPT-4o tokenizer): {len(encoder4o.encode(text))}
    """
    return print(message)

def analyze_chunks(chunks, use_tokens=False):
    dprint(f"{'='*33} def analyze_chunks(chunks, use_tokens={use_tokens}) {'='*33}")

    if len(chunks) < 2:
        return
    CHUNK1 = 0
    CHUNK2 = 0
    # Print the chunks of interest
    print("\nNumber of Chunks:", len(chunks))
    print("\n", "="*50, f"{CHUNK1}th Chunk", "="*50,"\n", chunks[CHUNK1])
    if use_tokens:
        count_tokens2(chunks[CHUNK1])
    print("\n", "="*50, f"{CHUNK2}st Chunk", "="*50,"\n", chunks[CHUNK2])
    if use_tokens:
        count_tokens2(chunks[CHUNK2])
    # chunk1, chunk2 = chunks[199], chunks[200]
    chunk1, chunk2 = chunks[CHUNK1], chunks[CHUNK2]
    
    if use_tokens:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens1 = encoding.encode(chunk1)
        tokens2 = encoding.encode(chunk2)
        

        # Find overlapping tokens
        for i in range(len(tokens1), 0, -1):
            if tokens1[-i:] == tokens2[:i]:
                overlap = encoding.decode(tokens1[-i:])
                print("\n", "="*50, f"\nOverlapping text ({i} tokens):", overlap)
                return
        print("\nNo token overlap found")
    else:
        # Find overlapping characters
        for i in range(min(len(chunk1), len(chunk2)), 0, -1):
            if chunk1[-i:] == chunk2[:i]:
                print("\n", "="*50, f"\nOverlapping text ({i} chars):", chunk1[-i:])
                return
        print("\nNo character overlap found")


def extract_embedded_files(docx_path, output_dir):
    dprint(f"{'='*33} def extract_embedded_files(docx_path, output_dir) {'='*33}")

    extracted_files = []
    # Open the .docx file as a zip archive
    with zipfile.ZipFile(docx_path, 'r') as docx_zip:
        # Look for embedded files in the 'word/embeddings' directory
        embeddings_dir = 'word/embeddings/'
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract each embedded file to the output directory
        for file in docx_zip.namelist():
            if file.startswith(embeddings_dir):
                docx_zip.extract(file, output_dir)
                extracted_files.append(file)
                dprint(f"Extracted: {file}")

    return extracted_files

def read_full_document_md(file_path):
    dprint(f"{'='*33} def read_full_document_md(file_path) {'='*33}")

    output_dir = 'extracted_files/'
    embeddings_dir = 'word/embeddings/'
    full_embed_path = f"{output_dir}/{embeddings_dir}"
    # extracted_files = extract_embedded_files(file_path, "extracted_files")
    extracted_files = extract_embedded_files(file_path, output_dir)
    document_text = convert_docx_to_markdown(file_path)
    # document_text = read_docx_in_order(file_path)
    if extracted_files:
        for i, extracted_file_path in enumerate(extracted_files, start=1):
            dprint('read_full_document() - iterating embedded files')
            dprint(f"extracted_file_path : {extracted_file_path}")
            if extracted_file_path.endswith(".docx"):
                embedded_path = os.path.join(output_dir, extracted_file_path)
                appended = convert_docx_to_markdown(embedded_path)
                document_text = f"{document_text} \n# Додаток №{i}: \n {appended}"
            # if extracted_file_path.endswith(".docx"):
            #     # document_text = f"{document_text} \n# Додаток №{i}: \n {read_docx_in_order(f"{output_dir}{extracted_file_path}")}"
            #     document_text = f"{document_text} \n# Додаток №{i}: \n {convert_docx_to_markdown(f"{output_dir}{extracted_file_path}")}"

        # for filename in os.listdir(f"{output_dir}/{embeddings_dir}"):
        for filename in os.listdir(full_embed_path):
            file_path = os.path.join(full_embed_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                dprint(f"Removed embedded file: {filename}")
    
    return document_text


def recursive_tokens_chunking(document):
    dprint(f"{'='*33} def recursive_tokens_chunking(document) {'='*33}")
    blocks = _split_markdown_into_blocks(document)
    chunks = _build_chunks(blocks)
    if DEBUG:
        analyze_chunks(chunks, use_tokens=True)
    return chunks


def enrich_chunks_with_metadata(
    chunks,
    document_text,
    encoding_model,
    filename,
    headings_lists=None,
    heading_paths=None,
    extra_metadata: Dict = None,
):
    dprint("enrich_chunks_with_metadata()")

    positions = _stable_find_positions(document_text, chunks)

    # try to get file stats (best-effort)
    file_size = None
    file_mtime_iso = None
    try:
        p = filename if os.path.isabs(filename) else os.path.join(os.getcwd(), filename)
        st = os.stat(p)
        file_size = st.st_size
        from datetime import datetime
        file_mtime_iso = datetime.fromtimestamp(st.st_mtime).isoformat()
    except Exception:
        pass

    enriched = []
    for i, chunk in enumerate(chunks):
        start, end = positions[i]
        token_count_o200k = _count_tokens(chunk)
        token_count_35 = len(_CL100K_ENCODER.encode(chunk, disallowed_special=()))
        word_count = len(chunk.split())
        headings_in_chunk = headings_lists[i] if headings_lists else []
        heading_path = heading_paths[i] if heading_paths else []
        section_title = heading_path[-1] if heading_path else (headings_in_chunk[-1] if headings_in_chunk else None)

        def _line_at(pos: int) -> int:
            if pos is None or pos < 0:
                return None
            return document_text.count("\n", 0, pos) + 1

        meta = {
            "chunk_id": str(uuid.uuid4()),
            "doc_id": _sha1(filename),
            "checksum_sha1": _sha1(chunk),

            "chunk_index": i,
            "char_range": [start, end],
            "line_span": [_line_at(start), _line_at(end)],

            "token_count_o200k": token_count_o200k,
            "token_count_cl100k": token_count_35,
            "word_count": word_count,
            "chunk_token_count": token_count_o200k,

            "headings": headings_in_chunk,
            "heading_path": heading_path,
            "section_title": section_title,

            "filename": filename,
            "file_size": file_size,
            "file_mtime_iso": file_mtime_iso,
            "encoding_model": encoding_model,
        }

        # serialize list fields for vectorstores that reject non-scalars
        meta["char_range_str"] = json.dumps(meta["char_range"], ensure_ascii=False)
        meta["line_span_str"] = json.dumps(meta["line_span"], ensure_ascii=False)
        
        if extra_metadata:
            # do not overwrite the core keys unless explicitly provided
            for k, v in extra_metadata.items():
                meta.setdefault(k, v)

        enriched.append({
            "text": chunk,
            "metadata": meta
        })
    return enriched



def convert_docx_to_markdown(docx_path: str) -> str:
    print(f"{'='*33} def convert_docx_to_markdown(docx_path: str) -> str: {'='*33}")

    return pypandoc.convert_file(docx_path, 'markdown', format='docx')


def get_chunks_with_meta_md_only(file_path):
    dprint(f"{'='*33} def get_chunks_with_meta_md_only(file_path) {'='*33}")

    with open(file_path, "r", encoding="utf-8") as md_file:
        md_document_text = md_file.read()
    chunks = recursive_tokens_chunking(md_document_text)

    info = _extract_heading_path_for_chunks(chunks)
    headings_lists = [d["headings_in_chunk"] for d in info]
    heading_paths = [d["heading_path"] for d in info]

    extra_meta = _infer_attachment_metadata_from_md_path(file_path)

    enriched_chunks = enrich_chunks_with_metadata(
        chunks,
        md_document_text,
        ENCODING_MODEL,
        os.path.basename(file_path),
        headings_lists=headings_lists,
        heading_paths=heading_paths,
        extra_metadata=extra_meta,
    )
    return enriched_chunks


def get_chunks_from_all_documents_in_directory_md_only(documents_dir):
    dprint(f"{'='*33} def get_chunks_from_all_documents_in_directory_md_only(documents_dir) {'='*33}")
    """Recursively walk an MD root and return LangchainDocument chunks.

    Preserves metadata enrichment (including department inferred from path).
    """
    all_chunks = []
    for root, _dirs, files in os.walk(documents_dir):
        for filename in files:
            if not filename.lower().endswith(".md"):
                continue
            full_path = os.path.join(root, filename)
            enriched_chunks = get_chunks_with_meta_md_only(full_path)
            for chunk in enriched_chunks:
                all_chunks.append(LangchainDocument(
                    page_content=chunk["text"],
                    metadata=chunk["metadata"]
                ))
    return all_chunks



if __name__ == "__main__":
    print(f"{'='*33} if __name__ == __main__: parse_to_chunks.py {'='*33}")


    # file_path = 'documents/IRU2.docx'
    file_path = 'documents/_poryadok_koristuvannya_epb_v3-1_final.docx'

    enriched_chunks = get_chunks_with_meta(file_path)

    total_tokens = 0

    for i, chunk in enumerate(enriched_chunks):
        total_tokens += chunk['metadata']['token_count']
        print(Fore.BLUE + f"-----------------------------------------------------------------------------------------------enriched_chunk {i}---chunk_size: {chunk['metadata']['token_count']}" + Fore.RESET)

        print(chunk)
        print(Fore.BLUE + f"-----------------------------------------------------------------------------------------------end of enriched_chunk {i} ---chunk_size: {chunk['metadata']['token_count']}" + Fore.RESET)

    print(Fore.YELLOW + f"TOTAL TOKENS : {total_tokens}" + Fore.RESET)