import os
from langchain_core.vectorstores import VectorStore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGEngine, PGVectorStore
from file_reader import read_mutiple_pdfs
from dotenv import load_dotenv

load_dotenv()


embedding_model= NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b", truncate="END")

engine= PGEngine.from_connection_string(
    url= os.environ["DATABASE_URL"]
)

vector_store= PGVectorStore.create_sync(
    engine=engine, 
    table_name="pdf_documents",
    embedding_service=embedding_model
)

PDF_PATH= [
    'pdfs/Cau truc bao cao mau.pdf',
    'pdfs/python.pdf'
]


docs = read_mutiple_pdfs(PDF_PATH)
text_splitter= RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap= 200)
all_splits= text_splitter.split_documents(docs)

print(f"Split doc into {len(all_splits)} chunks")

for index, doc in enumerate(all_splits, start=1):
    print(f"Dang luu {index}/{len(all_splits)}")
    vector_store.add_documents(documents=[doc])

print(f"Indexed {len(all_splits)} chunks")