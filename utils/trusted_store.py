import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from utils.embedding_store import embeddings

TRUSTED_DB_FOLDER = "trusted_vectorstore"
TRUSTED_INDEX_NAME = "trusted_index"

persist_path = os.path.join(os.path.dirname(__file__), "..", "trusted_vectorstore")

def load_trusted_vectorstore():
    index_file = os.path.join(persist_path, "index.faiss")
    if not os.path.exists(index_file):
        return None  # trusted DB doesn't exist yet
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)


    # Dummy doc to initialize
    dummy_doc = [Document(page_content="dummy", metadata={"source": "init"})]
    return FAISS.from_documents(dummy_doc, embeddings)

def save_to_trusted_store(texts, metadatas=None):
    if not os.path.exists(persist_path):
        os.makedirs(persist_path)

    if os.path.exists(os.path.join(persist_path, "index.faiss")):
        db = FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
        db.add_texts(texts, metadatas)
    else:
        db = FAISS.from_texts(texts, embeddings, metadatas)
    db.save_local(persist_path)


def search_trusted_store(query, k=3):
    db = load_trusted_vectorstore()
    if db is None:
        return []  # nothing found if DB doesn't exist yet
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)

