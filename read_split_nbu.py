import os
import sys
import re
import json
from typing import List, Dict, Optional

from colorama import Fore

from docx import Document
from docx.oxml.ns import qn

# Debug gating via Config.DEBUG
try:
    from config import Config
    DEBUG = getattr(Config, "DEBUG", False)
except Exception:
    DEBUG = False

def dprint(msg: str):
    if DEBUG:
        print(msg)

def get_effective_font_size(run, para) -> Optional[float]:
    dprint("get_effective_font_size()")

    # Directly on the run
    try:
        if run.font.size:
            return run.font.size.pt
    except Exception:
        pass

    # Character style applied to run
    try:
        if run.style and run.style.font and run.style.font.size:
            return run.style.font.size.pt
    except Exception:
        pass

    # Paragraph style
    try:
        if para.style and para.style.font and para.style.font.size:
            return para.style.font.size.pt
    except Exception:
        pass

    # Default document style (Normal)
    try:
        normal_style = run.part.styles.get('Normal')
        if normal_style and normal_style.font and normal_style.font.size:
            return normal_style.font.size.pt
    except Exception:
        pass

    return None  # Unknown

def normalize_multiline_enumeration(text: str) -> str:
    dprint("normalize_multiline_enumeration()")

    # Крок 1: перетворити весь текст на список рядків
    lines = text.strip().splitlines()

    # Крок 2: проходимо по рядках, формуючи "поточний пункт"
    result_lines = []
    current = ''

    for line in lines:
        line = line.strip()

        if re.match(r'^\d+\)', line):  # новий пункт 1), 2), ..., 10)
            if current:
                result_lines.append(current.strip())
            current = line
        elif re.match(r'^\d+\.', line):  # наприклад: 117.
            if current:
                result_lines.append(current.strip())
            current = line
        else:
            current += ' ' + line  # додаємо до поточного пункту

    if current:
        result_lines.append(current.strip())

    # Крок 3: обʼєднати всі в один текст
    return '\n'.join(result_lines)

#TODO: доки в тексті не буде знайдено "I. Загальні положення" ми не записуємо текст
#TODO: потрібно знайти фінальну точку документу, після якої буде зупинено запис доку

# --- Paragraph ID helpers ---
def get_paragraph_id(para, fallback_idx: Optional[int] = None) -> str:
    dprint("get_paragraph_id()")

    """Return Word's paragraph ID (w14:paraId) if present, else a stable fallback.
    The fallback uses the paragraph index if provided, otherwise a hash of text.
    """
    el = para._element
    pid = el.get(qn('w14:paraId')) or el.get(qn('w:paraId'))
    if pid:
        return pid
    # Fallback: deterministic id from index if available, else from text
    if fallback_idx is not None:
        return f"p{fallback_idx:06d}"
    # last resort – avoid very long ids
    return f"p_{abs(hash(para.text)) % (10**12)}"

def extract_text_with_font(docx_path: str) -> str:
    dprint("extract_text_with_font()")

    doc = Document(docx_path)
    full_text = []

    start_write_flag = False

    for i, para in enumerate(doc.paragraphs, 1):
        # print(f"para {i}:" + "-"*80)
        # print(para.text)
        # print("non strip up| strip down"+"-"*20)
        # print(para.text.strip())
        # print("-"*80)

        paragraph_text = ' '.join(para.text.split())
        # print(paragraph_text)
        for run in para.runs:
            text = run.text.strip()
            # print(f"run.text:" + "-"*40)
            # print(text)
            # print("-"*40)
            # print(f" text.find('I. Загальні положення') ---- {text.find('I. Загальні положення')}")
            if not text:
                continue
            if text.find('I. Загальні положення') != -1:
                start_write_flag = True
            if text.startswith('{') and text.endswith('}'):
                dprint(f"Skipping placeholder text: {text}")
                continue

            is_bold = run.bold
            if is_bold:
                font_size = get_effective_font_size(run, para)
                if font_size and font_size > 12:
                    # print(f"WE HAVE TEXT_SIZE > 12 - BULLSHIT!!! font_size: {font_size} Text: {text}")
                    continue
                else:
                    # paragraph_text += text run.text.strip()
                    paragraph_text = paragraph_text.replace(run.text.strip(), '')
            else: 
                # paragraph_text += text
                pass

        if start_write_flag:
            if paragraph_text:
                if paragraph_text.startswith('{') and paragraph_text.endswith('}'):
                    # print(f"WE HAVE {{ }} - BULLSHIT!!! Text: {paragraph_text}")
                    continue
                else:
                    # print("paragraph_text"+"-"*80)
                    # print(paragraph_text)
                    # # print("="*40)
                    # # print(paragraph_text.strip())
                    # print("-"*80)
                    full_text.append(paragraph_text.strip())

    complete_text = "\n".join(full_text)
    return complete_text

# --- Extract text with paragraph IDs ---
def extract_text_with_font_and_ids(docx_path: str) -> List[Dict]:
    dprint("extract_text_with_font_and_ids()")

    doc = Document(docx_path)
    items = []  # list of {text, para_id}
    start_write_flag = False

    for i, para in enumerate(doc.paragraphs, 1):
        paragraph_text = ' '.join(para.text.split())
        for run in para.runs:
            text = run.text.strip()
            if not text:
                continue
            if text.find('I. Загальні положення') != -1:
                start_write_flag = True
            if text.startswith('{') and text.endswith('}'):
                # skip placeholders
                continue
            is_bold = run.bold
            if is_bold:
                font_size = get_effective_font_size(run, para)
                if font_size and font_size > 12:
                    # skip large bold headings
                    continue
                else:
                    paragraph_text = paragraph_text.replace(run.text.strip(), '')

        if start_write_flag and paragraph_text:
            if paragraph_text.startswith('{') and paragraph_text.endswith('}'):
                continue
            pid = get_paragraph_id(para, fallback_idx=i)
            items.append({
                'text': paragraph_text.strip(),
                'para_id': pid,
                'para_index0': i - 1  # zero-based index into doc.paragraphs
            })

    return items

# def split_docx_to_question(file_path, second_split = False):
#     print(f"{'='*33}  {'='*33}")

#     readed_doc = extract_text_with_font(file_path)
#     pattern = r'(?s)((?:\d+(?:-\d+)?\.)\s.*?)(?=\n\d+(?:-\d+)?\.\s|\Z)'
#     questions = re.findall(pattern, readed_doc, re.DOTALL | re.MULTILINE)

#     ready_question = []
#     second_split_ready_question = []
    
#     for question in questions:
#         ready_question.append(normalize_multiline_enumeration(question))

#     if second_split:
#         second_split_pattern = r'(?s)((?:\d+(?:\.\d+)?\.)\s.*?)(?=\n\d+(?:\.\d+)?\.\s|\Z)'
#         for question in ready_question:
#             second_split_questions = re.findall(second_split_pattern, question, re.DOTALL | re.MULTILINE)
#             for ssq in second_split_questions:
#                 second_split_ready_question.append(ssq)

#     return second_split_ready_question if second_split else ready_question

def split_docx_to_question_with_ids(file_path: str, second_split: bool = False) -> List[Dict]:
    dprint("split_docx_to_question_with_ids()")

    """
    Split into numbered questions and keep a list of paragraph IDs per question.
    Returns a list of dicts: {'question_text': str, 'para_ids': [str, ...], 'para_indices0': [int, ...]}.
    """
    items = extract_text_with_font_and_ids(file_path)

    # Top-level markers like "117." or "308-1."
    top_marker = re.compile(r'^\d+(?:-\d+)?\.\s')

    # Build top-level buckets from consecutive paragraphs
    buckets_parts = []  # each item: {'parts': [{'text', 'pid', 'idx0'}, ...]}
    current = {'parts': []}
    for it in items:
        txt = it['text']
        if top_marker.match(txt):
            if current['parts']:
                buckets_parts.append(current)
            current = {'parts': []}
        current['parts'].append({'text': txt, 'pid': it['para_id'], 'idx0': it.get('para_index0')})
    if current['parts']:
        buckets_parts.append(current)

    # Helper to turn parts -> bucket object
    def build_bucket(parts):
        question_text = normalize_multiline_enumeration('\n'.join(p['text'] for p in parts))
        return {
            'question_text': question_text,
            'para_ids': [p['pid'] for p in parts],
            'para_indices0': [p['idx0'] for p in parts],
            '_parts': parts,   # keep for optional second split
        }

    buckets = [build_bucket(b['parts']) for b in buckets_parts]

    if not second_split:
        # Strip helper key before returning
        for b in buckets:
            b.pop('_parts', None)
        return buckets

    # Second-level markers like "1.1.", "2.3.4." (at least one dot level)
    sub_marker_block = re.compile(
        r'(?s)((?:\d+(?:\.\d+)+\.)\s.*?)(?=(?:\n\d+(?:\.\d+)+\.\s)|\Z)'
    )

    refined = []
    for b in buckets:
        parts = b['_parts']
        # Build joined text and paragraph spans (start,end) for overlap calc
        joined = ''
        spans = []  # list of (start, end, pid, idx0)
        pos = 0
        for p in parts:
            t = p['text']
            start = pos
            joined += t
            pos += len(t)
            end = pos
            spans.append((start, end, p['pid'], p['idx0']))
            # add the newline we used when joining for detection between paragraphs
            joined += '\n'
            pos += 1

        matches = list(re.finditer(sub_marker_block, joined))

        if not matches:
            # No sub-splits detected — keep the whole bucket
            refined.append({
                'question_text': b['question_text'],
                'para_ids': b['para_ids'],
                'para_indices0': b['para_indices0'],
            })
            continue

        # Build refined sub-questions with their OWN para_ids by overlap
        for m in matches:
            mstart, mend = m.span()
            sub_ids, sub_idxs = [], []
            for s, e, pid, idx in spans:
                # overlap if paragraph span intersects match span
                if s < mend and e > mstart:
                    sub_ids.append(pid)
                    sub_idxs.append(idx)
            refined.append({
                'question_text': normalize_multiline_enumeration(m.group(1)),
                'para_ids': sub_ids,
                'para_indices0': sub_idxs,
            })

    return refined



# def print_runs_for_para_ids(docx_path, para_ids):
#     print(f"{'='*33}  {'='*33}")

#     doc = Document(docx_path)
#     for para in doc.paragraphs:
#         pid = para._element.get(qn('w14:paraId')) or para._element.get(qn('w:paraId'))
#         # print(f"PID : {pid}")
#         if pid in para_ids:
#             print(f"--- Paragraph ID: {pid} ---")
#             for run in para.runs:
#                 print(f"Run text: '{run.text}'")
#                 # print(f"Run text: '{run.text}' | Bold: {run.bold} | Italic: {run.italic}")


# Print runs for given paragraph IDs; if IDs are missing in the document, fall back to indices.
# para_indices0 are zero-based indices into doc.paragraphs.
def print_runs_for_paras(docx_path, para_ids=None, para_indices0=None):
    dprint("print_runs_for_paras()")

    """Print runs for given paragraph IDs; if IDs are missing in the document, fall back to indices.
    para_indices0 are zero-based indices into doc.paragraphs.
    """
    doc = Document(docx_path)

    # Fast path: if indices provided, use them directly
    if para_indices0:
        for idx in para_indices0:
            if 0 <= idx < len(doc.paragraphs):
                para = doc.paragraphs[idx]
                pid = para._element.get(qn('w14:paraId')) or para._element.get(qn('w:paraId'))
                if DEBUG:
                    print(f"--- Paragraph idx0={idx} pid={pid} ---")
                    print(para.text)
                # for run in para.runs:
                    # print(f"Run text: '{run.text}'")
        return

    # Build map from IDs to paragraph
    if para_ids:
        id_to_para = {}
        for para in doc.paragraphs:
            pid = para._element.get(qn('w14:paraId')) or para._element.get(qn('w:paraId'))
            if pid:
                id_to_para[pid] = para
        for pid in para_ids:
            para = id_to_para.get(pid)
            if not para:
                continue
            if DEBUG:
                print(f"--- Paragraph ID: {pid} ---")
                print(para.text)
            # for run in para.runs:
            #     print(f"Run text: '{run.text}'")


if __name__ == "__main__":
    print(f"{'='*33} if __name__ == __main__: read_split_nbu.py {'='*33}")


    if len(sys.argv) > 1:
        file_name = sys.argv[1]

    nbu_folder_path = 'NBU/'

    # file_name = 'NBU64.docx'

    file_path = os.path.join(nbu_folder_path, file_name)

    # questions = split_docx_to_question(file_path, True)
    questions_struct = split_docx_to_question_with_ids(file_path, True)

    doc = Document(file_path)

    for i, q in enumerate(questions_struct):
        print(f"q['para_ids'] : {q['para_ids']}")
        print(f"q['para_indices0'] : {q.get('para_indices0')}")
        # Prefer IDs if the doc has them; otherwise fall back to indices
        print_runs_for_paras(file_path, para_ids=q['para_ids'], para_indices0=q.get('para_indices0'))
        # print(q)
        # print(q['para_ids'])
        # for id in q['para_ids']:
        #     print(f"paragraph id : {id}")
        #     print(doc.paragraphs[id])
        #     print("-"*50)
        if i > 10:
            break

    # Write plain text file (back-compat)
    with open(f"questions_new_gpt_{file_name[:-5]}.txt", "w", encoding='utf-8') as f:
        for q in questions_struct:
            f.write(q['question_text'])
            f.write("\n")
            f.write("-"*120)
            f.write("\n")

    # Write JSON with paragraph IDs for precise mapping back to the docx
    with open(f"questions_new_gpt_{file_name[:-5]}_with_ids.json", "w", encoding='utf-8') as jf:
        json.dump(questions_struct, jf, ensure_ascii=False, indent=2)

    print(f"\nlen of questions : {len(questions_struct)}")
