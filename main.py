import json
import time
import os
import argparse
from pdf_parser import EnhancedPDFParser
from analysis_engine import PersonaDrivenAnalyzer
from output_generator import generate_final_output

def load_challenge_input(input_path):
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Input file not found: {input_path}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Malformed JSON in input file: {input_path}")
        exit(1)

def resolve_pdf_path(pdf_path, input_json_path):
    base_dir = os.path.dirname(os.path.abspath(input_json_path))
    return os.path.join(base_dir, pdf_path)

def process_pipeline(task_context, documents, input_json_path, metadata):
    parser = EnhancedPDFParser()
    analyzer = PersonaDrivenAnalyzer()
    all_sections = []
    processed_docs = []
    for doc in documents:
        pdf_path = doc.get("path")
        if not pdf_path:
            filename = doc.get("filename")
            if filename:
                pdf_path = os.path.join('PDFs', filename)
        
        if not pdf_path:
            print(f"Missing PDF path or filename in document entry: {doc}")
            continue

        abs_pdf_path = resolve_pdf_path(pdf_path, input_json_path)
        if not os.path.exists(abs_pdf_path):
            print(f"Missing or invalid PDF path: {abs_pdf_path}")
            continue
        try:
            sections = parser.extract_structured_content(abs_pdf_path)
            for section in sections:
                section["document"] = os.path.basename(pdf_path)
            all_sections.extend(sections)
            processed_docs.append(doc)
        except Exception as e:
            print(f"Error parsing {abs_pdf_path}: {e}")
    metadata["documents"] = processed_docs
    ranked_sections, subsection_analyses = analyzer.analyze_document_collection(task_context, all_sections)
    return ranked_sections, subsection_analyses, metadata

def generate_output(results, output_path):
    ranked_sections, subsection_analyses, metadata = results
    generate_final_output(ranked_sections, subsection_analyses, metadata, output_path)

def main():
    parser = argparse.ArgumentParser(description="Persona-driven PDF analysis pipeline.")
    parser.add_argument('--input', type=str, default='challenge1b_input.json', help='Path to input JSON file')
    parser.add_argument('--output', type=str, default='challenge1b_output.json', help='Path to output JSON file')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    input_config = load_challenge_input(input_path)
    persona = input_config["persona"]["role"]
    job_task = input_config["job_to_be_done"]["task"]
    documents = input_config["documents"]
    task_context = f"As a {persona}, I need to {job_task}"
    metadata = {
        "persona": persona,
        "job_to_be_done": job_task,
        "documents": [] # Will be populated by the pipeline
    }
    start_time = time.time()
    results = process_pipeline(task_context, documents, input_path, metadata)
    elapsed = time.time() - start_time
    print(f"Processing completed in {elapsed:.2f} seconds.")
    generate_output(results, output_path)

if __name__ == "__main__":
    main()