import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"


def load_pdfs():
    documents = []

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(DATA_DIR, filename)
            loader = PyPDFLoader(path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = filename
                doc.metadata["page"] = doc.metadata.get("page", 0) + 1

            documents.extend(docs)

    return documents


def main():
    print("Chargement des PDF...")
    documents = load_pdfs()
    print(f"{len(documents)} pages chargées.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)
    print(f"{len(chunks)} chunks créés.")

    print("Création des embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Construction de l'index FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTORSTORE_DIR)
    print("Index FAISS sauvegardé dans vectorstore/")


if __name__ == "__main__":
    main()