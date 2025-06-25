from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

def clean_text(file_path):
    return Path(file_path).read_text(encoding="utf-8")

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def build_documents(chunks, region="India", theme="general", season="any"):
    return [
        Document(page_content=chunk, metadata={
            "region": region,
            "theme": theme,
            "season": season
        }) for chunk in chunks
    ]

def embed_documents():
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_files = Path("rag/clean").glob("*.txt")
    all_docs = []

    for file in text_files:
        text = clean_text(file)
        chunks = chunk_text(text)
        docs = build_documents(chunks)
        all_docs.extend(docs)

    vector_db = FAISS.from_documents(all_docs, embedder)
    vector_db.save_local("rag/vector_store")
    print("✅ Vector store created at: rag/vector_store")

if __name__ == "__main__":
    embed_documents()
