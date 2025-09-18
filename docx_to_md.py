import os
import sys
import zipfile
from typing import List, Optional

import pypandoc

# Debug gate
DEBUG = False
try:
    from config import Config  # optional
    DEBUG = getattr(Config, "DEBUG", False)
except Exception:
    pass

def dprint(msg: str):
    if DEBUG:
        print(msg)


def extract_embedded_files(docx_path: str, output_dir: str) -> List[str]:
    dprint(f"{'='*33} extract_embedded_files {'='*33}")
    extracted_files: List[str] = []
    with zipfile.ZipFile(docx_path, 'r') as docx_zip:
        embeddings_dir = 'word/embeddings/'
        os.makedirs(output_dir, exist_ok=True)
        for file in docx_zip.namelist():
            if file.startswith(embeddings_dir):
                docx_zip.extract(file, output_dir)
                extracted_files.append(file)
                dprint(f"Extracted: {file}")
    return extracted_files


def _ensure_pandoc_installed() -> None:
    """Verify the pandoc binary is available for pypandoc."""
    try:
        _ = pypandoc.get_pandoc_version()
    except Exception as e:
        raise RuntimeError(
            "Pandoc is required for DOCX->Markdown conversion. "
            "Install pandoc on the host (e.g., `yum install pandoc`) and ensure it is in PATH. "
            f"Details: {e}"
        )

def convert_docx_to_markdown(docx_path: str) -> str:
    dprint(f"{'='*33} convert_docx_to_markdown {'='*33}")
    try:
        _ensure_pandoc_installed()
        return pypandoc.convert_file(docx_path, 'markdown', format='docx')
    except Exception as e:
        # Provide a clearer error if pandoc isn't installed
        raise RuntimeError(f"Failed to convert '{docx_path}' to Markdown: {e}")


def read_full_document_md(file_path: str, save_folder_path: Optional[str] = None) -> str:
    dprint(f"{'='*33} read_full_document_md {'='*33}")
    output_dir = 'extracted_files/'
    embeddings_dir = 'word/embeddings/'
    full_embed_path = f"{output_dir}/{embeddings_dir}"
    extracted_files = extract_embedded_files(file_path, output_dir)
    document_text = convert_docx_to_markdown(file_path)
    if extracted_files:
        if not save_folder_path:
            # default output folder for dodatok markdown files
            save_folder_path = "documents/documents_converted_to_md/"
        os.makedirs(save_folder_path, exist_ok=True)
        for i, extracted_file_path in enumerate(extracted_files, start=1):
            dprint('Iterating embedded files')
            dprint(f"extracted_file_path : {extracted_file_path}")
            if extracted_file_path.endswith(".docx"):
                extracted_doc_text = convert_docx_to_markdown(f"{output_dir}{extracted_file_path}")
                file_name = os.path.basename(file_path)[:-5]
                out_path = os.path.join(save_folder_path, f"{file_name}_dodatok_{i}.md")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(extracted_doc_text)
                dprint(f"Saved dodatok to {out_path}")

        # Cleanup extracted embedded files
        if os.path.isdir(full_embed_path):
            for filename in os.listdir(full_embed_path):
                path = os.path.join(full_embed_path, filename)
                if os.path.isfile(path):
                    os.remove(path)
                    dprint(f"Removed {filename}")
    return document_text


def run_logic(files_path: str = '', save_folder_path: str = '') -> None:
    dprint(f"{'='*33} run_logic {'='*33}")
    """
    Recursively convert all .docx under `files_path` into Markdown under
    `save_folder_path`, preserving the relative subfolder structure.

    Also extracts embedded .docx as Markdown next to their parent file.
    This function is intended for full rebuilds and will clean the target
    output root before writing.
    """
    import shutil

    if not files_path:
        files_path = "documents/"
    if not save_folder_path:
        save_folder_path = "documents/documents_converted_to_md/"

    # Clean output folder (including subfolders) best-effort
    try:
        if os.path.isdir(save_folder_path):
            shutil.rmtree(save_folder_path)
    except Exception as e:
        print(f"Error cleaning {save_folder_path} directory. Details: {e}")
    os.makedirs(save_folder_path, exist_ok=True)

    # Walk all subfolders under files_path
    for root, _dirs, files in os.walk(files_path):
        rel_dir = os.path.relpath(root, files_path)
        out_dir = save_folder_path if rel_dir in (".", "") else os.path.join(save_folder_path, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        if DEBUG:
            dprint(f"Processing dir: {root} -> {out_dir}")
        for file in files:
            if not file.lower().endswith(".docx"):
                continue
            path_to_file = os.path.join(root, file)
            # Convert this .docx and write primary .md into mirrored out_dir
            markdown_text = read_full_document_md(path_to_file, out_dir)
            md_out_path = os.path.join(out_dir, f"{os.path.splitext(file)[0]}.md")
            with open(md_out_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            dprint(f"Converted {path_to_file} -> {md_out_path}")


if __name__ == "__main__":
    dprint(f"{'='*33} docx_to_md.py {'='*33}")
    if len(sys.argv) > 1:
        document_path = sys.argv[1]
        print(f"File to work with: {document_path}")
        if document_path.lower().endswith(".docx"):
            markdown_text = read_full_document_md(document_path)
            with open(f"{document_path[:-4]}md", "w", encoding="utf-8") as f:
                f.write(markdown_text)
    else:
        print("No argument was added. Working with default folder")
        run_logic()

