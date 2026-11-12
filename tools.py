import uuid
from deepagents.backends import StateBackend
from langchain.tools import tool
from indexing import vector_store

backend = StateBackend() # tao cac file ao de agent doc, khong luu that vao o cung 

@tool(parse_docstring=True) #lay mo ta ham va tham so tu doc_string de mo ta tool 
def search_documentation(query:str)-> str:
    """
    Search LangChain documentation and save matching chunks to the agent filesystem.

    Args:
        query: NL search query.
    
    Returns:
        File path where retrieved chunks were saved under /retrieved/.
    """

    retrieved_docs= vector_store.similarity_search(query=query, k=3,) #lay tu db ra context
    batch_id= uuid.uuid4().hex[:8] #tao id rieng 8 ki tu de phan biet cac lan tim kiem 
    uploads: list[tuple[str, bytes]]= [] 
    save_paths: list[str] = []

    for index, doc in enumerate(retrieved_docs, start=1):
        path= f"/retrieved/{batch_id}/chunk_{index}.md"
        content= (
            f"# Source: {doc.metadata.get('source', 'unknown')}\n\n"
            f"{doc.page_content}"
        )
        uploads.append((path, content.encode("utf-8")))
        save_paths.append(path)

    backend.upload_files(uploads)
    return(
        f"saved {len(save_paths)} documentation chunks: \n"
        + "\n".join(save_paths)
    )
