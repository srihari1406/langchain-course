import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("/Users/srihari/Documents/Agentic_AI/Learning_Materials/langchain-course/data/mediumblog1.txt",autodetect_encoding=True)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")

    print("Creating vector store...")
    PineconeVectorStore.from_documents(split_docs, embeddings, index_name=os.getenv("INDEX_NAME"))
    print("Done!")