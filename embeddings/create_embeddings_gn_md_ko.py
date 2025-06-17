# create_embeddings_gn_md_ko.py

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from pathlib import Path
import os

def load_documents_from_folder(folder_path):
    documents = []
    for file_path in Path(folder_path).glob("*.md"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents.extend(loader.load())
    return documents

def main():
    data_path = "docx_ko/Glumerulonephritis"  # Glomerulonephritis 요약문서 경로
    vector_store_path = "vector_store_gn_ko"

    if not os.path.exists(data_path):
        raise ValueError("GN 요약문서 디렉토리를 찾을 수 없습니다.")

    print("📂 Loading documents...")
    documents = load_documents_from_folder(data_path)

    print("✂️ Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print("📦 Creating FAISS vector store...")
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(vector_store_path)

    print(f"✅ GN 벡터 저장 완료: {vector_store_path}")

if __name__ == "__main__":
    main()
