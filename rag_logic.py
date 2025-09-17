import os
import sys
import time
import json
import logging
import datetime

from langchain.schema import Document as LangchainDocument
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

from typing import Any, Dict

from dotenv import load_dotenv

from openai import RateLimitError  # Import RateLimitError for retry logic

from config import Config

# from parse import parse_docx_to_chunks, parse_docx_to_chunks_md
from parse_to_chunks import (
    get_chunks_from_all_documents_in_directory_md_only,
    get_chunks_with_meta_md_only,
)

from docx_to_md import run_logic as convert_docx_to_markdown

from rag_question import question_logic
from rag_document import document_logic


os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
# Log to stdout to avoid unbounded file growth
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='{"t":"%(asctime)s","level":"%(levelname)s","msg":%(message)r}',
    )

### 25000 tokens per minute is MAX rn

REBUILD_DB = Config.REBUILD_DB


# GPT_MODEL = 'gpt-4o'
# GPT_MODEL = 'gpt-4.1'
# if GPT_MODEL == 'gpt-4.1':
#     DEPLOYMENT_NAME = os.getenv("GPT_DEPLOYMENT_NAME_4_1")
# elif GPT_MODEL == 'gpt-4o':
#     DEPLOYMENT_NAME = os.getenv("GPT_DEPLOYMENT_NAME_4o")
# else:
#     DEPLOYMENT_NAME = os.getenv("GPT_DEPLOYMENT_NAME")

# print(f"DEPLOYMENT_NAME : {DEPLOYMENT_NAME}")


DEBUG = getattr(Config, "DEBUG", False)

def dprint(msg: str):
    if DEBUG:
        print(msg)



# CREATE_NEW_RESULT_TXT = True
# START_QUESTION = 0 #47 # TODO: for continue from last error session.
# QUESTION_SKIP_LIST = [] #[44]
# CONTINUE_RESULT_TXT_NAME = "result_of_NBU(NBU95)_to_VND_comparison_25-07-2025 16:10:04.txt"

# FOR TRACING TPR LIMITS ON AZURE MODEL:
# MAX_TOKENS_PER_MINUTE = 200000 #25000
# TOKEN_WINDOW_SECONDS = 60

def get_llm(kind: str = 'document'):
    dprint(f"{'='*33} def get_llm(kind='document'): {'='*33}")


    if kind == 'document':
        return AzureChatOpenAI(
            # deployment_name="gpt-4o-ub-test-080624",   # Your Azure deployment name
            # deployment_name=os.getenv(DEPLOYMENT_NAME),   # Your Azure deployment name
            deployment_name=Config.DEPLOYMENT_NAME,   # Your Azure deployment name
            model_name=Config.GPT_MODEL,                       # Model type you selected in Azure
            openai_api_version="2024-02-01",
            azure_endpoint=os.getenv("GPT_ENDPOINT"),
            api_key=os.getenv("GPT_KEY"),

            # 🔧 Model behavior tuning
            response_format={"type": "json_object"},
            temperature=0.0,          # Low = more factual, ideal for RAG
            max_tokens=4096,           # Max output tokens. Fits most document Q&A responses
            top_p=0.9,                # Controls diversity (keep default)
            frequency_penalty=0.0,    # Avoids penalizing repeated terms (safe for legal)
            presence_penalty=0.0,     # Don't discourage reuse of key phrases
            request_timeout=120,       # Allow longer responses for large prompts
        )
    if kind == 'question':
        return AzureChatOpenAI(
            # deployment_name="gpt-4o-ub-test-080624",   # Your Azure deployment name
            deployment_name=Config.DEPLOYMENT_NAME,   # Your Azure deployment name
            model_name=Config.GPT_MODEL,                       # Model type you selected in Azure
            openai_api_version="2024-02-01",
            azure_endpoint=os.getenv("GPT_ENDPOINT"),
            api_key=os.getenv("GPT_KEY"),

            # 🔧 Model behavior tuning
            response_format={"type": "json_object"},
            temperature=0.3,          # Low = more factual, ideal for RAG
            max_tokens=4096,           # Max output tokens. Fits most document Q&A responses
            top_p=0.9,                # Controls diversity (keep default)
            frequency_penalty=0.0,    # Avoids penalizing repeated terms (safe for legal)
            presence_penalty=0.0,     # Don't discourage reuse of key phrases
            request_timeout=60,       # Avoid long hangs
        )


def get_embedding_model():
    dprint(f"{'='*33} def get_embedding_model(): {'='*33}")


    return AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-large",  # Your Azure deployment name
        model="text-embedding-3-large",             # Model name used in Azure
        api_key=os.getenv("TE3L_KEY"),
        azure_endpoint=os.getenv("TE3L_ENDPOINT"),
        openai_api_version="2024-02-01"
    )


def should_rebuild_vectorstore():
    dprint(f"{'='*33} def should_rebuild_vectorstore(): {'='*33}")


    target = Config.DB_NAME
    if not os.path.exists(target):
        return True
    if os.path.isdir(target):
        # if the dir exists but has no contents, consider it missing
        if not any(os.scandir(target)):
            return True
    return Config.REBUILD_DB


def _sanitize_metadata(meta: Any) -> Dict[str, Any]:
    dprint(f"{'='*33} def _sanitize_metadata(meta: Any) -> Dict[str, Any]: {'='*33}")


    """
    Chroma accepts only str/int/float/bool/None.
    Coerce everything else to JSON strings. If meta isn't a dict, return {}.
    """
    if not isinstance(meta, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            try:
                out[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                # last resort string cast
                out[k] = str(v)
    return out

def _normalize_documents(objs):
    dprint(f"{'='*33} def _normalize_documents(objs): {'='*33}")


    """
    Ensure we end up with a list[LangchainDocument].
    Accepts LangchainDocument, dict-like with 'page_content', or raw strings.
    """
    norm = []
    for i, d in enumerate(objs):
        try:
            if isinstance(d, LangchainDocument):
                norm.append(
                    LangchainDocument(
                        page_content=d.page_content,
                        metadata=_sanitize_metadata(d.metadata)
                    )
                )
            elif isinstance(d, dict) and "page_content" in d:
                norm.append(
                    LangchainDocument(
                        page_content=d.get("page_content", ""),
                        metadata=_sanitize_metadata(d.get("metadata", {}))
                    )
                )
            elif isinstance(d, str):
                norm.append(
                    LangchainDocument(
                        page_content=d,
                        metadata={}
                    )
                )
            else:
                print(f"⚠️ Skipping unsupported chunk type at index {i}: {type(d)}")
        except Exception as e:
            print(f"⚠️ Failed to normalize chunk at index {i}: {e}")
    return norm


def rebuild_vector_store():
    dprint(f"{'='*33} def rebuild_vector_store(): {'='*33}")
    logging.info("Rebuilding vectorstore (file-by-file)…")
    BATCH_SIZE = Config.EMBED_BATCH_SIZE or 5
    db_path = Config.DB_NAME
    docs_root = Config.DOCS_PATH
    md_root = Config.DOCS_PATH_MD
    os.makedirs(docs_root, exist_ok=True)

    # Fresh rebuild: clear existing DB dir; also reset MD mirror to avoid stale files
    if os.path.exists(db_path) and os.path.isdir(db_path):
        import shutil
        shutil.rmtree(db_path)
    if os.path.isdir(md_root):
        try:
            import shutil
            shutil.rmtree(md_root)
        except Exception:
            pass
    os.makedirs(md_root, exist_ok=True)

    # Single embedding client and collection for the whole run
    embedding_model = get_embedding_model()
    db = Chroma(
        collection_name=Config.COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=db_path
    )

    total_files = 0
    for dirpath, _dirs, files in os.walk(docs_root):
        for name in files:
            if not name.lower().endswith('.docx'):
                continue
            full = os.path.join(dirpath, name)
            total_files += 1
            try:
                upsert_ird_document(full, db=db)
                logging.info(f"Upserted {os.path.relpath(full, docs_root)}")
            except RateLimitError as e:
                logging.warning(f"Rate limit on {name}; waiting 60s and retrying once… ({e})")
                time.sleep(60)
                upsert_ird_document(full, db=db)
            except Exception:
                logging.exception(f"Failed to upsert {full}")
                raise

    try:
        db.persist()
    except Exception:
        pass
    logging.info(f"Rebuild complete. Files processed: {total_files}")
    return db


def _open_db_for_update() -> Chroma:
    """Open existing Chroma collection with embeddings for updates."""
    # Ensure parent directory exists for the persisted DB
    try:
        parent = os.path.dirname(getattr(Config, 'DB_NAME', 'vectorstore/db'))
        if parent:
            os.makedirs(parent, exist_ok=True)
    except Exception:
        pass
    embedding_model = get_embedding_model()
    db = Chroma(
        collection_name=Config.COLLECTION_NAME,
        persist_directory=Config.DB_NAME,
        embedding_function=embedding_model,
    )
    return db


def upsert_ird_document(docx_path: str, db: Chroma | None = None) -> None:
    """Rebuild chunks for a single IRD (.docx) and update the vector store.

    - Preserves department by mirroring the doc's relative folder into MD output.
    - Deletes previous vectors for this IRD by attachment_group_id (parent stem sha1).
    - Adds fresh chunks from the main MD and all its `_dodatok_*.md` attachments.
    """
    dprint(f"{'='*33} upsert_ird_document {'='*33}")
    if not docx_path.lower().endswith('.docx'):
        raise ValueError("docx_path must point to a .docx file")

    # Compute MD output dir preserving relative path from DOCS_PATH
    src_root = Config.DOCS_PATH
    md_root = Config.DOCS_PATH_MD
    rel = os.path.relpath(docx_path, src_root)
    rel_dir = os.path.dirname(rel)
    base_stem = os.path.splitext(os.path.basename(docx_path))[0]
    out_dir = os.path.join(md_root, rel_dir) if rel_dir not in (".", "") else md_root
    os.makedirs(out_dir, exist_ok=True)

    # Clean old MD for this doc (main + dodatok_*.md) to avoid stale content
    try:
        for name in os.listdir(out_dir):
            if not name.lower().endswith('.md'):
                continue
            if name == f"{base_stem}.md" or name.startswith(f"{base_stem}_dodatok_"):
                try:
                    os.remove(os.path.join(out_dir, name))
                except FileNotFoundError:
                    pass
    except FileNotFoundError:
        pass

    # Convert this DOCX (writes dodatok md's in out_dir), write main md
    md_text = convert_docx_to_markdown.read_full_document_md(docx_path, out_dir) if hasattr(convert_docx_to_markdown, 'read_full_document_md') else None
    if md_text is None:
        # If run_logic was passed instead, call the module function directly
        from docx_to_md import read_full_document_md as _rf
        md_text = _rf(docx_path, out_dir)
    main_md_path = os.path.join(out_dir, f"{base_stem}.md")
    with open(main_md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    # Collect MD files belonging to this IRD: main + dodatok_*.md
    md_files = []
    for name in os.listdir(out_dir):
        if not name.lower().endswith('.md'):
            continue
        if name == f"{base_stem}.md" or name.startswith(f"{base_stem}_dodatok_"):
            md_files.append(os.path.join(out_dir, name))

    # Chunk all related MD files
    from langchain.schema import Document as LCDoc
    docs = []
    for p in sorted(md_files):
        enriched = get_chunks_with_meta_md_only(p)
        for ch in enriched:
            # Chroma only accepts scalar metadata; sanitize complex types
            safe_meta = _sanitize_metadata(ch.get("metadata", {}))
            docs.append(LCDoc(page_content=ch.get("text", ""), metadata=safe_meta))

    if not docs:
        print(f"⚠️ No chunks found for {docx_path}")
        return

    # Compute path-safe group id: sha1(relative_docx_path_without_extension)
    import hashlib
    # Prefer content hash as stable id
    def _sha256_file(p: str) -> str | None:
        try:
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                for chunk in iter(lambda: f.read(1024*1024), b''):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None
    content_hash = _sha256_file(docx_path)
    if content_hash:
        group_id = content_hash
    else:
        # Fallbacks for robustness
        rel_docx = os.path.relpath(docx_path, Config.DOCS_PATH)
        rel_no_ext = os.path.splitext(rel_docx.replace("\\", "/"))[0]
        group_id = hashlib.sha1(rel_no_ext.encode("utf-8")).hexdigest()

    # Open DB and replace this IRD's vectors (reuse provided db if any)
    if db is None:
        db = _open_db_for_update()
    # Also try legacy ids deletion to avoid stale duplicates during migration
    try:
        legacy_gid = hashlib.sha1(base_stem.encode("utf-8")).hexdigest()
        try:
            db.delete(where={"attachment_group_id": legacy_gid})
        except Exception:
            pass
        # And legacy path-based id
        try:
            rel_docx = os.path.relpath(docx_path, Config.DOCS_PATH)
            rel_no_ext = os.path.splitext(rel_docx.replace("\\", "/"))[0]
            legacy_path_gid = hashlib.sha1(rel_no_ext.encode("utf-8")).hexdigest()
            db.delete(where={"attachment_group_id": legacy_path_gid})
        except Exception:
            pass
        db.delete(where={"attachment_group_id": group_id})
    except Exception as e:
        print(f"ℹ️ Delete by attachment_group_id failed or nothing to delete: {e}")

    # Deterministic ids per chunk to avoid duplicates
    ids = []
    for d in docs:
        did = d.metadata.get("doc_id") or ""
        cidx = d.metadata.get("chunk_index")
        sha = (d.metadata.get("checksum_sha1") or "")[:8]
        ids.append(f"{did}:{cidx}:{sha}")

    # Embed in smaller batches with basic backoff
    BATCH_SIZE = getattr(Config, 'EMBED_BATCH_SIZE', 5) or 5
    for i in range(0, len(docs), BATCH_SIZE):
        b_docs = docs[i:i+BATCH_SIZE]
        b_ids = ids[i:i+BATCH_SIZE]
        success = False
        delay = 2.0
        attempts = 0
        while not success:
            try:
                db.add_documents(b_docs, ids=b_ids)
                db.persist()
                success = True
            except RateLimitError as e:
                attempts += 1
                if attempts >= 5:
                    raise
                logging.warning(f"Embedding rate-limited (batch {i}-{i+len(b_docs)}). Retry in {int(delay)}s… ({e})")
                time.sleep(delay)
                delay = min(delay * 2, 60)
            except Exception:
                raise
    print(f"✅ Upserted IRD: {docx_path} ({len(docs)} chunks)")


def main_logic(mode, nbu_document_name=''):
    dprint(f"{'='*33} def main_logic(mode, nbu_document_name=''): {'='*33}")

    dprint(f"REBUILD_DB : {REBUILD_DB}")
    dprint("Waiting 2 seconds...")
    time.sleep(2)
    dprint("Continue to run RAG solution...")


    # if not os.path.exists(doc_path):
    if not os.path.exists(Config.DOCS_PATH):
        print(f"ERROR: File not found — {Config.DOCS_PATH}")
        exit(1)

    if should_rebuild_vectorstore():
        db = rebuild_vector_store()
    else:
        logging.info(
            "Loading existing vectorstore… persist_directory=%s embedding_model=%s",
            Config.DB_NAME, Config.EMBEDDING_MODEL
        )

        embedding_model = get_embedding_model()
        db = Chroma(
            collection_name=Config.COLLECTION_NAME,
            persist_directory=Config.DB_NAME,
            embedding_function=embedding_model
        )
        try:
            count = db._collection.count()
            logging.info(f"Loaded collection size from '{Config.DB_NAME}': {count}")
            try:
                probe = db.similarity_search("bank", k=1)
                logging.debug(f"Probe search returned {len(probe)} doc(s).")
            except Exception as e:
                print(f"ℹ️ Probe search failed: {e}")
        except Exception as e:
            logging.warning(f"Error in db._collection.count(): {e}")

    llm = get_llm(mode)

    # Mode selecting. Chat or document.
    if mode == 'question':
        question_logic(db, llm)
    elif mode == 'document':
        document_logic(db, llm, nbu_document_name)

    # nbu_document_name = 'NBU95.docx'
    ct = datetime.datetime.now()
    fct = ct.strftime("%d-%m-%Y %H:%M:%S")


        


# main_logic("document", 'NBU95.docx')

if __name__ == "__main__":
    logging.info(f"{'='*33} if __name__ == __main__: rag_logic.py {'='*33}")


    if len(sys.argv) > 1:
        logging.info("argv=%s", sys.argv)
        mode = sys.argv[1]
        logging.info(f"Selected | {mode} | mode")
        if mode == 'ingest':
            # CLI: python rag_logic.py ingest documents/<dept>/file.docx
            try:
                docx_path = sys.argv[2]
            except Exception as e:
                logging.error(f"Error: provide a .docx path under {Config.DOCS_PATH}. Details: {e}")
                sys.exit(1)
            if not os.path.isabs(docx_path):
                docx_path = os.path.join(os.getcwd(), docx_path)
            upsert_ird_document(docx_path)
        else:
            try:
                document_name = sys.argv[2]
                logging.info(f" File to work with : {document_name}")
                if document_name.endswith(".docx"):
                    main_logic(mode, document_name)
            except Exception as e:
                document_name = None
                logging.warning(f"Error: second argument are not provided. Details: {e}")
            
            if not document_name:
                main_logic(mode)

    else:
        logging.info("No argument was added. Closing the app.")
        exit
