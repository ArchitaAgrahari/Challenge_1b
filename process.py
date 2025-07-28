import os
import json
import fitz  # PyMuPDF
import re
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

########## CONFIGURATION (EDIT HERE AS NEEDED) ##########
COLLECTION_NAME = "Collection 1"  # Change as needed!
BASE_DIR = "CHALLENGE_1B"         # <-- match screenshot
COLLECTION_DIR = os.path.join(BASE_DIR, COLLECTION_NAME)
PDF_DIR = os.path.join(COLLECTION_DIR, "PDFs")
OUTLINE_DIR = COLLECTION_DIR
SUBSECTION_FILE = os.path.join(COLLECTION_DIR, "subsection_analysis.json")
INPUT_JSON_PATH = os.path.join(COLLECTION_DIR, "challenge1b_input.json")
OUTPUT_JSON_PATH = os.path.join(COLLECTION_DIR, "challenge1b_output.json")

NUMERIC_HEADING_RE = re.compile(r"^(\d+(\.\d+)*)(\.|\s)\s*(.*)")

def clean(text): return re.sub(r'\s+', ' ', text.strip())
def get_dot_level(num): return num.count('.') + 1 if num else 1

def extract_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    blocks, font_sizes, freq_counter = [], [], Counter()
    for page_index, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans: continue
                text = clean(" ".join(span["text"] for span in spans if span["text"].strip()))
                if not text: continue
                size = max(span["size"] for span in spans)
                fonts = [span["font"] for span in spans]
                is_bold = any("bold" in f.lower() for f in fonts)
                is_italic = any("italic" in f.lower() or "oblique" in f.lower() for f in fonts)
                y0 = spans[0]["bbox"][1] if spans else 0
                blocks.append({
                    "text": text, "size": size, "page": page_index, "is_bold": is_bold,"is_italic": is_italic,"y0": y0
                })
                font_sizes.append(size)
                freq_counter[text.lower()] += 1
    return blocks, font_sizes, freq_counter

def is_heading_candidate(text, size, body_size, is_bold, is_italic):
    if not text or len(text) < 4 or len(text) > 120: return False
    if size < body_size + 1 and not NUMERIC_HEADING_RE.match(text): return False
    if text.isupper() and not NUMERIC_HEADING_RE.match(text): return False
    if any(x in text.lower() for x in ["version", "copyright", "all rights reserved"]): return False
    return is_bold or is_italic or NUMERIC_HEADING_RE.match(text)

def classify_level(text):
    m = NUMERIC_HEADING_RE.match(text)
    if m:
        num = m.group(1); level = get_dot_level(num)
        return f"H{level}"
    return None

def extract_headings(blocks, font_sizes):
    body_size = Counter(font_sizes).most_common(1)[0][0] if font_sizes else 12
    outline, seen, title_candidate = [], set(), None
    for block in blocks:
        text, page, size = block["text"].rstrip(), block["page"], block["size"]
        is_bold, is_italic = block["is_bold"], block["is_italic"]
        text_key = (text.lower(), page)
        if text_key in seen: continue
        seen.add(text_key)
        if title_candidate is None and page == 0 and size > body_size + 1 and \
           not NUMERIC_HEADING_RE.match(text) and not is_heading_candidate(text, size, body_size, is_bold, is_italic):
            title_candidate = text; continue
        if title_candidate and text == title_candidate and page == 0: continue
        if not is_heading_candidate(text, size, body_size, is_bold, is_italic): continue
        level = classify_level(text)
        if not level:
            diff = size - body_size
            level = "H1" if diff > 2 else ("H2" if diff > 1.2 else "H3")
        outline.append({"level": level, "text": text + (" " if not text.endswith(" ") else ""), "page": page})
    outline = sorted(outline, key=lambda h: (h["page"], next(b["y0"] for b in blocks if b["text"] == h["text"] and b["page"] == h["page"])))
    return title_candidate.strip() if title_candidate else "", outline

def process_pdf_for_1a(pdf_path, output_json_path):
    blocks, font_sizes, freq_counter = extract_blocks(pdf_path)
    title, outline = extract_headings(blocks, font_sizes)
    data = {"title": title, "outline": outline}
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def batch_process_1a():
    print(f"Running Round 1A on PDFs in: {PDF_DIR}")
    os.makedirs(OUTLINE_DIR, exist_ok=True)
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        out_json = os.path.join(OUTLINE_DIR, pdf_file.rsplit('.', 1)[0] + ".json")
        print(f"Extracting outline for {pdf_file} to {out_json}")
        try: process_pdf_for_1a(pdf_path, out_json)
        except Exception as e: print(f"Error processing {pdf_file}: {e}")

def load_outline(pdf_file):
    base_name = os.path.splitext(pdf_file)[0]
    outline_path = os.path.join(OUTLINE_DIR, base_name + ".json")
    if not os.path.exists(outline_path):
        print(f"Outline JSON {outline_path} missing for {pdf_file}. Returning empty outline.")
        return {"title": base_name, "outline": []}
    with open(outline_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_section_text(pdf_path, headings):
    doc = fitz.open(pdf_path)
    headings_sorted = sorted(headings, key=lambda h: (h["page"], h["text"]))
    sections = []
    for i, h in enumerate(headings_sorted):
        start_page = h["page"] - 1
        heading_text = h["text"]
        section_content = ""
        for p in range(start_page, doc.page_count):
            page = doc.load_page(p)
            page_text = page.get_text("text")
            if p == start_page:
                idx = page_text.lower().find(heading_text.lower())
                if idx != -1:
                    page_text = page_text[idx + len(heading_text):]
            if i + 1 < len(headings_sorted) and headings_sorted[i+1]["page"] - 1 == p:
                next_heading = headings_sorted[i+1]["text"].lower()
                next_idx = page_text.lower().find(next_heading)
                if next_idx != -1:
                    page_text = page_text[:next_idx]
            section_content += page_text
            if i + 1 < len(headings_sorted):
                next_page_idx = headings_sorted[i+1]["page"] - 1
                if p == next_page_idx - 1: break
            else:
                if p == doc.page_count - 1: break
        sections.append({
            "level": h["level"],
            "text": heading_text,
            "page": h["page"],
            "content": section_content.strip()
        })
    return sections

def generate_subsection_analysis():
    print(f"Generating subsection analysis json for collection {COLLECTION_NAME}")
    all_sections = []
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        outline = load_outline(pdf_file)
        headings = outline.get("outline", [])
        if not headings: continue
        try:
            sections = extract_section_text(pdf_path, headings)
            for sec in sections:
                all_sections.append({
                    "document": pdf_file,
                    "section_title": sec["text"],
                    "page_number": sec["page"],
                    "refined_text": sec["content"]
                })
        except Exception as e:
            print(f"Error extracting sections from {pdf_file}: {e}")
    with open(SUBSECTION_FILE, "w", encoding="utf-8") as f:
        json.dump(all_sections, f, indent=2, ensure_ascii=False)
    print(f"Saved subsection analysis to {SUBSECTION_FILE}")

def main():
    batch_process_1a()
    generate_subsection_analysis()
    # --- 1B ranking and output ---
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    persona = input_data.get("persona", {}).get("role", "Unknown Persona")
    job = input_data.get("job_to_be_done", {}).get("task", "")
    pdf_documents = input_data.get("documents", [])
    # Load subsection_analysis.json with normalization of keys
    refined_text_map = {}
    if os.path.exists(SUBSECTION_FILE):
        with open(SUBSECTION_FILE, "r", encoding="utf-8") as f:
            refined_list = json.load(f)
            for entry in refined_list:
                key = (entry.get("document", "").strip().lower(), entry.get("section_title", "").strip().lower())
                refined_text_map[key] = entry.get("refined_text", "")
    # Initialize encoder
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    query_embedding = model.encode([f"{persona}. {job}"])[0]
    all_sections = []
    for doc_info in pdf_documents:
        pdf_file = doc_info["filename"]
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        if not os.path.exists(pdf_path): continue
        outline = load_outline(pdf_file)
        headings = outline.get("outline", [])
        if not headings: continue
        sections = extract_section_text(pdf_path, headings)
        for sec in sections:
            combined_text = sec["text"] + ". " + (sec["content"][:1000] if sec["content"] else "")
            sec_emb = model.encode([combined_text])[0]
            sim_score = float(cosine_similarity([query_embedding], [sec_emb])[0][0])
            key = (pdf_file.strip().lower(), sec["text"].strip().lower())
            refined = refined_text_map.get(key, "")
            all_sections.append({
                "document": pdf_file,
                "section_title": sec["text"],
                "page_number": sec["page"],
                "similarity": sim_score,
                "level": sec["level"],
                "refined_text": refined
            })
    all_sections_sorted = sorted(all_sections, key=lambda x: -x["similarity"])
    top_sections = all_sections_sorted[:5]
    output_json = {
        "metadata": {
            "input_documents": [d["filename"] for d in pdf_documents],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.utcnow().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }
    for idx, sec in enumerate(top_sections, start=1):
        output_json["extracted_sections"].append({
            "document": sec["document"],
            "section_title": sec["section_title"],
            "importance_rank": idx,
            "page_number": sec["page_number"]
        })
        if sec["refined_text"] and sec["refined_text"].strip():
            output_json["subsection_analysis"].append({
                "document": sec["document"],
                "refined_text": sec["refined_text"],
                "page_number": sec["page_number"]
            })
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as outf:
        json.dump(output_json, outf, indent=2, ensure_ascii=False)
    print(f"Round 1B output saved to: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
