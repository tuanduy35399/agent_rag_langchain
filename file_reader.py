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

def read_mutiple_pdfs(pdf_path: List[str | Path]) -> List[Document]: 
    """
    read text from mutiple pdfs and return a list of content
    """
    docs: List[Document] = []
    for path in pdf_path:
        path_obj = Path(path)
        text= read_pdf_text(path_obj)
        docs.append(Document(page_content=text, metadata={"source": str(path_obj)}))
    return docs


# Test 2 func
# PDF_PATH= [
#     'pdfs/Cau truc bao cao mau.pdf',
#     'pdfs/python.pdf'
# ]

# print(read_mutiple_pdfs(PDF_PATH))