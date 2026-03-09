import getpass
import logging
import os
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from logging import basicConfig, INFO
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  

load_dotenv()

basicConfig(level=INFO)
logger = logging.getLogger(__name__)

vector_store = None
agent = None


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    global vector_store
    retrieved_docs = vector_store.similarity_search(query, k=2)
    logger.info(f"Retrieved {(retrieved_docs)} documents for the query: '{query}'")
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def generate_embeddings(text):
    global vector_store, agent
    if not text or not text.strip():
        return "No text provided."
    llm = ChatOpenAI(
            model=os.getenv("CHAT_MODEL"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )

    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"),
                                  base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"))
            
    vector_store = InMemoryVectorStore(embeddings)

    logger.info("Generating embeddings for the provided text...")
    
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
    
    all_splits = splitter.split_text(text)
    logger.info(f"Text split into {len(all_splits)} chunks.")
    logger.info(f"First 3 chunks: {all_splits[:3]}")    
    # logger.info(all_splits)
    
    document_ids = vector_store.add_texts(all_splits)
    logger.info(print(document_ids[:3]))
    logger.info("Embeddings generated and stored successfully.")
    logger.info(f"Vector store now contains {(vector_store)} documents.")
    tools = [retrieve_context]
    prompt =""" You are a helpful RAG assistant.
    Use the retrieval tool to fetch relevant context before answering.
    Base your answer on the retrieved context.
    """
    agent = create_agent(
        model=llm,
        tools=tools,    
        system_prompt=prompt
    )

    return "Embeddings generated and RAG agent created successfully."

def answer_question(query: str) -> str:
    global agent

    if agent is None:
        return "Please build the RAG system first."

    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )

    return result["messages"][-1].content
