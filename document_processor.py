import PyPDF2
from pathlib import Path
from typing import Optional


def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extract text content from a PDF file
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string, or None if extraction fails
    """
    try:
        text_content = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if text.strip():
                    text_content.append(text)
        
        # Join all pages with double newline
        full_text = "\n\n".join(text_content)
        
        if not full_text.strip():
            print("Warning: No text could be extracted from PDF")
            return None
        
        return full_text
    
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return None


def chunk_text(text: str, max_chunk_size: int = 4000) -> list[str]:
    """
    Split text into manageable chunks for processing
    
    Args:
        text: The text to chunk
        max_chunk_size: Maximum size of each chunk
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for space
        
        if current_size + word_size > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks