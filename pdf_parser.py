import fitz  # PyMuPDF

class EnhancedPDFParser:
    def __init__(self):
        self.heading_indicators = {
            'font_size_jump': 2.0,      # Font size increase threshold
            'bold_weight': 600,         # Bold font weight
            'isolation_score': 0.8,     # Standalone text blocks
            'position_patterns': ['center', 'left_margin']
        }

    def extract_structured_content(self, pdf_path):
        sections = []
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if self._is_section_heading(block, blocks):
                    section = self._extract_section_content(page, block, blocks)
                    # Try to extract a meaningful section title
                    title = section["title"]
                    # --- ADVANCED HEADING EXTRACTION LOGIC ---
                    if not title or title.lower().startswith("paragraph") or len(title.split()) < 2:
                        # Try to extract heading from block itself (font size, bold, all-caps, etc.)
                        heading_candidate = self._extract_heading_from_block(block)
                        if heading_candidate:
                            title = heading_candidate
                        else:
                            # Try to extract from content lines
                            lines = [l.strip() for l in section["content"].split("\n") if l.strip()]
                            # Prefer all-caps or title-cased, short lines (likely dish names, headings, etc.)
                            for l in lines:
                                if (l.isupper() or l.istitle()) and 1 <= len(l.split()) <= 8 and not l.lower().startswith("paragraph"):
                                    title = l
                                    break
                            # Otherwise, look for lines that are bold or large font in the page blocks
                            if (not title or title.lower().startswith("paragraph")):
                                for b in blocks:
                                    if self._is_bold_or_styled(b):
                                        t = ''.join([span['text'] for line in b.get('lines', []) for span in line.get('spans', [])]).strip()
                                        if t and 1 <= len(t.split()) <= 8 and not t.lower().startswith("paragraph"):
                                            title = t
                                            break
                            # Otherwise, use first non-empty line
                            if (not title or title.lower().startswith("paragraph")) and lines:
                                title = lines[0][:40]
                            # Fallback: use first sentence
                            if not title or title.lower().startswith("paragraph"):
                                first_sentence = section["content"].split(". ")[0][:40]
                                if first_sentence:
                                    title = first_sentence.strip()
                    # Remove trailing colons, dashes, or numbering
                    if title:
                        import re
                        title = re.sub(r'^[0-9]+[.\-\)]\s*', '', title)
                        title = re.sub(r'[:\-\s]+$', '', title).strip()
                    sections.append({
                        "section_title": title,
                        "page_number": page_num + 1,
                        "content": section["content"],
                        "confidence_score": section["confidence"]
                    })
        # Only fallback if <2 real sections found
        meaningful_sections = [s for s in sections if s["section_title"] and not s["section_title"].lower().startswith("paragraph")]
        if len(meaningful_sections) < 2:
            # Try to extract better section titles from content before falling back
            sections = self._improve_section_titles(sections, doc)
            meaningful_sections = [s for s in sections if s["section_title"] and not s["section_title"].lower().startswith("paragraph")]
            if len(meaningful_sections) < 2:
                sections = self._fallback_paragraph_segmentation(doc)
        # Limit to 10 most confident, diverse sections (one per doc if possible)
        doc_seen = set()
        filtered = []
        for s in sorted(sections, key=lambda x: -x["confidence_score"]):
            doc_id = s.get("document")
            if doc_id not in doc_seen or len(filtered) < 5:
                filtered.append(s)
                if doc_id:
                    doc_seen.add(doc_id)
            if len(filtered) >= 10:
                break
        return filtered if filtered else sections

    def _improve_section_titles(self, sections, doc):
        """Try to extract better section titles from content"""
        improved_sections = []
        for section in sections:
            if section["section_title"] and not section["section_title"].lower().startswith("paragraph"):
                improved_sections.append(section)
                continue
            
            # Try to extract a better title from the content
            content = section["content"]
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            
            new_title = None
            # Look for lines that could be headings (short, capitalized, etc.)
            for line in lines[:5]:  # Check first 5 lines
                words = line.split()
                if 2 <= len(words) <= 10:  # Reasonable heading length
                    # Check if it looks like a heading
                    if (line.isupper() or line.istitle() or
                        any(word.isupper() for word in words[:3]) or
                        line.endswith(':') or
                        all(word[0].isupper() for word in words if word)):
                        new_title = line.rstrip(':').strip()
                        break
            
            # If still no good title, use first sentence
            if not new_title and lines:
                first_sentence = content.split('.')[0].strip()
                if 3 <= len(first_sentence.split()) <= 15:
                    new_title = first_sentence
                else:
                    new_title = lines[0][:50] + "..." if len(lines[0]) > 50 else lines[0]
            
            section["section_title"] = new_title or section["section_title"]
            improved_sections.append(section)
        
        return improved_sections

    def _extract_heading_from_block(self, block):
        # Try to extract a heading from the block using font size, bold, all-caps, etc.
        try:
            if not block.get('lines'):
                return None
            text = ''.join([span['text'] for line in block['lines'] for span in line['spans']]).strip()
            if not text:
                return None
            # Prefer all-caps or title-cased, short lines
            if (text.isupper() or text.istitle()) and 1 <= len(text.split()) <= 8:
                return text
            # Prefer bold or large font
            for line in block['lines']:
                for span in line['spans']:
                    if (span.get('flags', 0) & 2) or span.get('size', 0) > 14:
                        if 1 <= len(span['text'].split()) <= 8:
                            return span['text'].strip()
        except Exception:
            pass
        return None

    def _is_section_heading(self, block, context_blocks):
        score = 0
        if self._has_font_size_jump(block, context_blocks):
            score += 0.3
        if self._is_isolated_block(block):
            score += 0.2
        if self._is_bold_or_styled(block):
            score += 0.2
        if self._is_heading_length(block):
            score += 0.3
        return score > 0.6

    def _has_font_size_jump(self, block, context_blocks):
        try:
            block_size = block['lines'][0]['spans'][0]['size']
            sizes = [b['lines'][0]['spans'][0]['size'] for b in context_blocks if b.get('lines')]
            avg_size = sum(sizes) / len(sizes) if sizes else 0
            return block_size - avg_size > self.heading_indicators['font_size_jump']
        except Exception:
            return False

    def _is_isolated_block(self, block):
        # Heuristic: block with few lines and much whitespace
        try:
            if len(block.get('lines', [])) == 1 and block.get('width', 0) < 0.7 * block.get('bbox', [0,0,1000,0])[2]:
                return True
        except Exception:
            pass
        return False

    def _is_bold_or_styled(self, block):
        try:
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    if span.get('flags', 0) & 2:  # bold flag in PyMuPDF
                        return True
        except Exception:
            pass
        return False

    def _is_heading_length(self, block):
        try:
            text = ''.join([span['text'] for line in block.get('lines', []) for span in line.get('spans', [])])
            return 2 <= len(text.split()) <= 12
        except Exception:
            return False

    def _extract_section_content(self, page, heading_block, blocks):
        # Simple: get text from heading to next heading or end of page
        lines = []
        found_heading = False
        heading_text = ''
        for line in heading_block.get('lines', []):
            for span in line.get('spans', []):
                heading_text += span['text'] + ' '
        heading_text = heading_text.strip()
        for block in blocks:
            if found_heading:
                if self._is_section_heading(block, blocks):
                    break
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        lines.append(span['text'])
            elif block == heading_block:
                found_heading = True
        content = ' '.join(lines).strip()
        confidence = 0.9 if heading_text else 0.5
        return {"title": heading_text, "content": content, "confidence": confidence}

    def _fallback_paragraph_segmentation(self, doc):
        # Fallback: split by paragraphs or fixed chunking
        sections = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]
            for i, para in enumerate(paragraphs):
                # Try to extract a meaningful title from the paragraph
                lines = [l.strip() for l in para.split('\n') if l.strip()]
                title = f"Paragraph {i+1}"
                
                if lines:
                    # Look for a potential heading in the first few lines
                    for line in lines[:3]:
                        words = line.split()
                        if 2 <= len(words) <= 12:
                            # Check if it looks like a heading
                            if (line.isupper() or line.istitle() or
                                any(word.isupper() for word in words[:3]) or
                                line.endswith(':') or
                                all(word[0].isupper() for word in words if word)):
                                title = line.rstrip(':').strip()
                                break
                    
                    # If no heading found, use first sentence as title
                    if title.startswith("Paragraph"):
                        first_sentence = para.split('.')[0].strip()
                        if 3 <= len(first_sentence.split()) <= 15:
                            title = first_sentence
                        elif lines[0]:
                            title = lines[0][:60] + "..." if len(lines[0]) > 60 else lines[0]
                
                sections.append({
                    "section_title": title,
                    "page_number": page_num + 1,
                    "content": para,
                    "confidence_score": 0.4
                })
        if not sections:
            # Last resort: fixed chunking
            for page_num, page in enumerate(doc):
                text = page.get_text()
                chunk_size = 500
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i+chunk_size]
                    sections.append({
                        "section_title": f"Chunk {i//chunk_size+1}",
                        "page_number": page_num + 1,
                        "content": chunk,
                        "confidence_score": 0.2
                    })
        return sections
