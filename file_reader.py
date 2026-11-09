from pathlib import Path
from typing import List
from langchain_core.documents import Document
from pypdf import PdfReader

def read_pdf_text(pdf_path:str | Path) -> str:
    """
    read and return content in pdf file
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Pdf not found: {path}")

    reader=  PdfReader(str(path))
    text_parts= [page.extract_text() or "" for page in reader.pages]
    return "\n".join(text_parts)