import json
from datetime import datetime

def create_minimal_valid_output(metadata):
    return {
        "metadata": metadata,
        "extracted_sections": [],
        "subsection_analysis": []
    }

def generate_final_output(ranked_sections, subsection_analyses, metadata, output_path):
    max_sections = min(20, len(ranked_sections))
    
    # Extract just the filenames from the documents for the expected format
    input_documents = []
    for doc in metadata.get("documents", []):
        if isinstance(doc, dict):
            # If it's a dict with filename, extract just the filename
            input_documents.append(doc.get("filename", str(doc)))
        else:
            # If it's already a string, use it as is
            input_documents.append(str(doc))
    
    output_data = {
        "metadata": {
            "input_documents": input_documents,
            "persona": metadata.get("persona"),
            "job_to_be_done": metadata.get("job_to_be_done"),
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [
            {
                "document": section["document"],
                "section_title": section["section_title"],
                "importance_rank": i + 1,
                "page_number": section["page_number"]
            }
            for i, section in enumerate(ranked_sections[:max_sections])
        ],
        "subsection_analysis": subsection_analyses[:15]
    }
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        minimal_output = create_minimal_valid_output(metadata)
        with open(output_path, 'w') as f:
            json.dump(minimal_output, f, indent=2)
