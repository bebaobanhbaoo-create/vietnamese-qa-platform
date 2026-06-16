from typing import List, Dict
import re

class DocumentChunker:
    """Split documents into chunks with metadata tracking."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_tokens(self, text: str, doc_id: str) -> List[Dict]:
        """Split text into overlapping chunks by approximate token count."""
        words = text.split()
        chunks = []
        
        start = 0
        chunk_idx = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = ' '.join(words[start:end])
            
            chunks.append({
                'doc_id': doc_id,
                'chunk_id': f"{doc_id}_chunk_{chunk_idx}",
                'text': chunk_text,
                'start_idx': start,
                'end_idx': end
            })
            
            start = end - self.overlap if end < len(words) else end
            chunk_idx += 1
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str, doc_id: str) -> List[Dict]:
        """Split text by paragraphs, merging small ones."""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = []
        current_len = 0
        chunk_idx = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_len = len(para.split())
            
            if current_len + para_len > self.chunk_size and current_chunk:
                chunks.append({
                    'doc_id': doc_id,
                    'chunk_id': f"{doc_id}_chunk_{chunk_idx}",
                    'text': '\n\n'.join(current_chunk)
                })
                current_chunk = [para]
                current_len = para_len
                chunk_idx += 1
            else:
                current_chunk.append(para)
                current_len += para_len
        
        if current_chunk:
            chunks.append({
                'doc_id': doc_id,
                'chunk_id': f"{doc_id}_chunk_{chunk_idx}",
                'text': '\n\n'.join(current_chunk)
            })
        
        return chunks
