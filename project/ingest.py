from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

DATA_PATH = "data"
DB_PATH = "vectorstores/db_faiss"

def create_db():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)

if __name__ == "__main__":
    create_db()