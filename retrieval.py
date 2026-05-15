from operator import itemgetter
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

load_dotenv()


print("Initializing Components...")
embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

vectorStore = PineconeVectorStore(embedding=embeddings, index_name=os.getenv("INDEX_NAME"))
retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based on the following context:
    {context}
    Question: {question}
    Provide a detailed answer.""")
    

def format_doc(docs):
    """Formats retrieved documents into a string."""
    return "\n\n".join([doc.page_content for doc in docs])

# ======================================================================
# Implementation 2: Retrieval without LCEL
# ======================================================================
def retrieval_chain_without_lcel(query):
    """Retrieval without LCE."""
    retrieved_docs = retriever.invoke(query)
    context = format_doc(retrieved_docs)
    prompt = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content

# ======================================================================
# Implementation 3: Retrieval with LCEL
# ======================================================================
def retrieval_chain_with_lcel():
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") 
            | retriever 
            | format_doc
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")
    query = "What is Pinecone in machine learning?"

    #==============================================
    # Option 1: Raw invocation without RAG
    #==============================================
    print("\n" + "=" * 70)
    print("Implementation 1: Raw LLM Invocation without RAG")
    print("=" * 70 + "\n")
    raw_result = llm.invoke([HumanMessage(content=query)])
    print("Raw LLM Response:")
    print(raw_result.content)

    #==============================================
    # Option 2: Raw invocation without LCEL
    #==============================================
    print("\n" + "=" * 70)
    print("Implementation 2: Retrieval without LCEL")
    print("=" * 70 + "\n")
    retrieval_result = retrieval_chain_without_lcel(query)
    print("Retrieval without LCEL Response:")
    print(retrieval_result)

    #==============================================
    # Option 3: Raw invocation with LCEL
    #==============================================
    print("\n" + "=" * 70)
    print("Implementation 3: Retrieval with LCEL")
    print("=" * 70 + "\n")
    retrieval_chain = retrieval_chain_with_lcel()
    retrieval_with_lcel_result = retrieval_chain.invoke({"question": query})
    print("Retrieval with LCEL Response:")
    print(retrieval_with_lcel_result)