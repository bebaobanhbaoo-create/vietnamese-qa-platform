import os
from typing import Dict, Optional
import hashlib

class PDFParser:
    """Extract text content from PDF documents."""
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
    
    def parse(self, file_path: str) -> Optional[Dict]:
        """
        Parse PDF file and extract text content.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {ext}")
        
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(file_path)
            text_content = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    text_content.append({
                        'page': page_num + 1,
                        'content': text.strip()
                    })
            
            doc.close()
            
            full_text = '\n\n'.join([p['content'] for p in text_content])
            
            return {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'num_pages': len(text_content),
                'text': full_text,
                'pages': text_content,
                'hash': hashlib.md5(full_text.encode()).hexdigest()
            }
            
        except ImportError:
            raise ImportError("PyMuPDF (fitz) is required. Install with: pip install pymupdf")
    
    def batch_parse(self, directory: str) -> list:
        """Parse all PDF files in a directory."""
        results = []
        for filename in os.listdir(directory):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory, filename)
                try:
                    result = self.parse(file_path)
                    results.append(result)
                except Exception as e:
                    print(f"Error parsing {filename}: {e}")
        return results
