from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
import os, json

def extract_section_text(pdf_path, headings):
    doc = fitz.open(pdf_path)
    result = []
    for i, h in enumerate(headings):
        page = doc[h['page']-1]
        text = ""
        # Find heading position and extract until next heading
        # (You may improve chunking based on your 1A structure)
        text = page.get_text()  # Simplified: full text of the page
        result.append({'heading': h['text'], 'content': text, 'page': h['page'], 'level': h['level']})
    return result

# Load inputs
with open('challenge1b_input.json') as f:
    inp = json.load(f)
persona = inp['persona']
job = inp['job']
PDFS = inp['pdfs']

# Load or extract outlines from Round 1A outputs for each PDF
# For each PDF:
#   1. extract_section_text(pdf, outlines)
#   2. embed section content and title
#   3. embed persona+job, compute similarity, rank

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
query_emb = model.encode([f"{persona}. {job}"])[0]

results = []

for pdf_file in PDFS:
    headings = ... # Load from 1A output JSON for this PDF
    sections = extract_section_text(pdf_file, headings)
    section_texts = [f"{s['heading']}. {s['content']}" for s in sections]
    section_embs = model.encode(section_texts)
    sims = cosine_similarity([query_emb], section_embs)[0]
    ranked = sorted(zip(sections, sims), key=lambda x: -x[1])

    relevant = []
    for sec, sim in ranked[:3]:  # or use a threshold
        relevant.append({
            "level": sec['level'],
            "text": sec['heading'],
            "page": sec['page'],
            "similarity": float(sim)
        })
    results.append({"pdf": pdf_file, "relevant_sections": relevant})

output = {
    "persona": persona,
    "job": job,
    "results": results
}

with open('challenge1b_output.json', 'w', encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
