import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredFileLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ Singleton pattern to reuse embeddings everywhere
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_docs_and_store(input_folder, output_folder):
    docs = []

    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)

        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filename.endswith(".txt"):
                loader = TextLoader(filepath)
            elif filename.endswith(".pptx"):
                loader = UnstructuredPowerPointLoader(filepath)
            elif filename.endswith(".docx"):
                loader = UnstructuredFileLoader(filepath)
            else:
                continue

            docs.extend(loader.load())

        except Exception as e:
            print(f"⚠️ Skipping {filename}: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(output_folder, index_name="index")
