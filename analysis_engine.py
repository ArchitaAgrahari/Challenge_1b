from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)

class PersonaDrivenAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def analyze_document_collection(self, task_context, all_sections):
        task_embedding = self.model.encode([task_context])[0]
        persona_keywords = set([w.lower() for w in task_context.split() if len(w) > 3])
        doc_coverage = set()
        scored_sections = []
        for section in all_sections:
            heading_quality = 1.0 if section.get('section_title') and len(section['section_title'].split()) > 2 else 0.5
            doc_bonus = 0.2 if section['document'] not in doc_coverage else 0.0
            relevance_score = self._calculate_enhanced_relevance(task_embedding, section)
            final_score = relevance_score * 0.7 + heading_quality * 0.2 + doc_bonus * 0.1
            section['relevance_score'] = final_score
            scored_sections.append(section)
        # Prefer sections from diverse documents in top ranks
        ranked_sections = sorted(scored_sections, key=lambda x: x['relevance_score'], reverse=True)
        top_sections = []
        seen_docs = set()
        for sec in ranked_sections:
            if len(top_sections) >= 20:
                break
            if sec['document'] not in seen_docs or len(top_sections) < 10:
                top_sections.append(sec)
                seen_docs.add(sec['document'])
        # Subsection analysis: prefer contextually relevant sentences
        subsection_analyses = []
        for section in top_sections[:15]:
            refined_text = self._extract_key_sentences(task_embedding, section['content'], persona_keywords)
            subsection_analyses.append({
                "document": section['document'],
                "refined_text": refined_text,
                "page_number": section['page_number']
            })
        return top_sections, subsection_analyses

    def _calculate_enhanced_relevance(self, task_embedding, section):
        content_embedding = self.model.encode([section['content']])[0]
        semantic_score = cosine_similarity([task_embedding], [content_embedding])[0][0]
        confidence_weight = section.get('confidence_score', 0.5)
        length_score = min(len(section['content']) / 1000, 1.0)
        final_score = (
            semantic_score * 0.7 +
            confidence_weight * 0.2 +
            length_score * 0.1
        )
        return final_score

    def _extract_key_sentences(self, task_embedding, section_content, persona_keywords=None):
        sentences = sent_tokenize(section_content)
        if len(sentences) <= 3:
            return section_content
        sentence_embeddings = self.model.encode(sentences)
        sentence_scores = []
        for i, sent_emb in enumerate(sentence_embeddings):
            similarity = cosine_similarity([task_embedding], [sent_emb])[0][0]
            position_boost = 0.1 if i == 0 or i == len(sentences)-1 else 0
            keyword_bonus = 0.15 if persona_keywords and any(kw in sentences[i].lower() for kw in persona_keywords) else 0
            score = similarity + position_boost + keyword_bonus
            sentence_scores.append((i, score, sentences[i]))
        if len(sentences) > 10:
            top_k = 5
        elif len(sentences) > 5:
            top_k = 3
        else:
            top_k = 2
        selected = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:top_k]
        selected = sorted(selected, key=lambda x: x[0])
        return ' '.join([sent[2] for sent in selected])
