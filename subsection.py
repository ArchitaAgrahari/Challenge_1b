import os
import json
import fitz  # PyMuPDF
import re
from tqdm import tqdm
#from sentence_transformers import SentenceTransformer  # Uncomment if using summarization
# from sklearn.metrics.pairwise import cosine_similarity

# CONFIG -- change accordingly
COLLECTION_DIR = r"Challenge_1b/Collection_1"   # Change your collection folder
PDF_DIR = os.path.join(COLLECTION_DIR, "PDFs")
OUTLINE_DIR = COLLECTION_DIR
INPUT_JSON_PATH = os.path.join(COLLECTION_DIR, "challenge1b_input.json")  # adjust if different
OUTPUT_JSON_PATH = os.path.join(COLLECTION_DIR, "subsection_analysis.json")

# Summarization toggle
ENABLE_SUMMARIZATION = False

# --- Optional Summarizer Setup (you can skip if ENABLE_SUMMARIZATION = False) ---
'''
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
def summarize(text):
    # truncate to 1024 tokens approx for BART input length limit
    max_len = 1024
    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len]
    # generate summary
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]["summary_text"]
'''

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_outline(json_path):
    if not os.path.isfile(json_path):
        print(f"Outline json not found: {json_path}")
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_section_text(pdf_path, headings):
    doc = fitz.open(pdf_path)
    extracted_sections = []

    # Sort headings by page and text occurrence (already expected sorted)
    headings = sorted(headings, key=lambda h: (h['page'], h['text']))

    for idx, heading in enumerate(headings):
        start_page_idx = heading['page'] - 1
        heading_text = heading['text']

        section_text = ""

        # Compile remaining headings to know where to stop
        if idx + 1 < len(headings):
            next_heading = headings[idx + 1]
            next_page_idx = next_heading['page'] - 1
            next_heading_text = next_heading['text']
        else:
            next_heading = None

        for p in range(start_page_idx, doc.page_count):
            page = doc[p]
            page_text = page.get_text("text")

            if p == start_page_idx:
                # Start from after the heading text occurs on this page
                start_pos = page_text.lower().find(heading_text.lower())
                if start_pos >= 0:
                    page_text = page_text[start_pos + len(heading_text):]

            # If the next heading is on the same page, truncate text before next heading
            if next_heading and p == next_page_idx:
                next_pos = page_text.lower().find(next_heading_text.lower())
                if next_pos >= 0:
                    page_text = page_text[:next_pos]

            section_text += page_text

            # Stop at page before next heading page
            if next_heading and p == next_page_idx - 1:
                break

        extracted_sections.append({
            "heading": heading_text,
            "page": heading['page'],
            "text": clean_text(section_text)
        })

    return extracted_sections

def main():
    # Load input JSON specifying documents and persona/job
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        input_json = json.load(f)

    pdf_files = [doc['filename'] for doc in input_json.get('documents', [])]

    subsection_data = []

    # If summarization enabled, uncomment accordingly
    # model = SentenceTransformer("paraphrase-MiniLM-L6")
    # But simplified here
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        outline_json_path = os.path.join(OUTLINE_DIR, os.path.splitext(pdf_file)[0] + '.json')

        outline_json = load_outline(outline_json_path)
        if outline_json is None or not outline_json.get('outline'):
            print(f"No outline for {pdf_file}, skipping...")
            continue

        headings = outline_json['outline']

        # Extract chunks of text for sections
        sections = extract_section_text(pdf_path, headings)

        for section in sections:
            text_to_use = section['text']
            # Optionally summarize
            if ENABLE_SUMMARIZATION:
                # text_to_use = summarize(text_to_use)  # Uncomment summarizer usage if desired
                pass

            # Build entry dictionary
            entry = {
                "document": pdf_file,
                "section_title": section['heading'],
                "page": section['page'],
                "refined_text": text_to_use
            }
            subsection_data.append(entry)

    # Save JSON file
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as outf:
        json.dump(subsection_data, outf, indent=2, ensure_ascii=False)

    print(f"Subsection analysis JSON saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
