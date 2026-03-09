import streamlit as st
from agent import run_agent
from rag import generate_embeddings, answer_question
import logging
from logging import basicConfig, INFO
basicConfig(level=INFO)
logger = logging.getLogger(__name__)
vector_store = None


st.set_page_config(page_title="Business Model Generator", page_icon="🤖😎")

st.title("🤖 Agentic AI Business Model Generator")

tab1, tab2 = st.tabs(["Business Model Generator", "RAG Q&A"])

with tab1:
    st.header("Business Model Generator")
    st.write("Enter a business idea, and the AI agent will generate a business model for it.")
    
    business_idea = st.text_area(
        "Enter your business idea:",
        placeholder="Example: A subscription-based meal kit delivery service that focuses on healthy, locally sourced ingredients.",
    )
    output = "" 
    if st.button("Generate Business Model", key="generate_bm"):
        if business_idea.strip():
            with st.spinner("Generating business model..."):
                try:
                    output = run_agent(business_idea)
                    st.subheader("Generated Business Model")
                    st.divider()
                    st.markdown(output)
                    st.session_state["generated_output"] = output
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
        else:
            st.warning("Please enter a business idea first.")
    if "generated_output" in st.session_state:
        st.divider()
        st.subheader("Generated Business Model")
        st.markdown(st.session_state["generated_output"])
    
with tab2:
    st.header("RAG Question Answering")
    st.write("Paste text, build the RAG system, and ask questions about it.")
    try:
        generate_embeddings(output)
        st.success("RAG system built successfully! You can now ask questions about the text.")
    except Exception as e:
        st.error(f"Something went wrong while building the RAG system: {e}")

    st.divider()
    rag_query = st.text_input(
        "Ask a question about the text:",
        placeholder="Example: What is the revenue model?",
        key="rag_query"
    )
    if st.button("Ask Question", key="ask_rag"):
        if rag_query.strip():
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    response = answer_question(rag_query)
                    st.subheader("Answer")
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Something went wrong while answering: {e}")
        else:
            st.warning("Please enter a question first.")