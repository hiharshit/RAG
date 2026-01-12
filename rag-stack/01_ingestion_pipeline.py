import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv


load_dotenv()


def load_document(docs_path="docs"):
    """Load all the text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    # check if docs dirctory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory {docs_path} does not exist. Please create it and add your documents"
        )

    # load all the .txt files from the documents directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No .txt files found in {docs_path}. Please add your documents"
        )
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into smaller chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks, persist_directory="db/chorma_db"):
    """Create and persisst ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB")

    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # create ChromaDB vector store
    print("--- Creating ChromaDB vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print("--- Finished creating vector store ---")

    print(f"Vector store created and sved to {persist_directory}")
    return vectorstore


def main():
    """Main ingestion pipeline"""

    # define paths
    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    # check if vector store already exists
    if os.path.exists(persistent_directory):
        print("Vector store exists already, no need to reprocess the documents.")
        embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )
        print(
            f"Loading existing vector store with {vectorstore._collection.count()} documents."
        )
        return vectorstore

    # load the files
    documents = load_document(docs_path)

    # chunkign the files
    chunks = split_documents(documents=documents)

    # embedding and storing in the vector database
    vectorstore = create_vector_store(
        chunks=chunks, persist_directory=persistent_directory
    )


if __name__ == "__main__":
    main()
