import os 
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from rag import generate_embeddings
llm = ChatOpenAI(
            model=os.getenv("CHAT_MODEL"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def generate_context(query: str) -> str:
    """Retrieve supporting business context for a startup idea."""
    return f"Relevant context for {query}: subscription revenue, digital acquisition, customer retention, and MVP prioritization."

tools = [search, generate_context]
agent = create_agent(
    model=llm,
    tools=tools,
)
print("Agent created successfully.")


def run_agent(user_input):
    system_prompt= f"""
You are a startup advisor.

Generate a clear business model for the following idea.

Business Idea: {user_input}

Return the response in this format:
***** Always generate in Document format so it can be read by langchain's text splitter so it can be used in a rag model*****

Business Idea Summary:
Problem:
Target Customers:
Value Proposition:
Revenue Model:
Distribution Channels:
Key Metrics:
MVP Features:
Risks:
Flow Chart: *** Pictorial representation of the business model, showing the flow from idea to customer acquisition and revenue generation. Use simple ASCII art or a textual description to illustrate the flow. ***
Go-to-Market Strategy:
"""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": system_prompt}]}
    )
    return result["messages"][-1].content


