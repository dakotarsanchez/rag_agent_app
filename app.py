import streamlit as st
import os
from your_rag_agent import DocumentAnalysisAgents  # Import your existing RAG agent

st.set_page_config(page_title="RAG Agent Interface", layout="wide")

st.title("RAG Agent Interface")

# Add proper error handling for API key
try:
    OPENAI_API_KEY = st.text_input("Enter your OpenAI API Key:", type="password")
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = str(OPENAI_API_KEY)  # Ensure string conversion
    else:
        st.warning("Please enter your OpenAI API key to continue.")

# Input area
query = st.text_area("Enter your query:", height=150)

# Optional file uploaders
meeting_notes = st.file_uploader("Upload Meeting Notes (optional)", type=["txt"])
client_agreements = st.file_uploader("Upload Client Agreements (optional)", type=["txt"])

if st.button("Run Query", type="primary"):
    if query:
        with st.spinner("Processing..."):
            try:
                # Initialize your RAG agent
                analysis_workflow = DocumentAnalysisAgents()
                
                # Process uploaded files if any
                meeting_notes_text = meeting_notes.getvalue().decode() if meeting_notes else ""
                client_agreements_text = client_agreements.getvalue().decode() if client_agreements else ""
                
                # Run analysis
                result = analysis_workflow.execute_analysis(
                    query=query,
                    meeting_notes=meeting_notes_text,
                    client_agreements=client_agreements_text,
                    client_id="P23"
                )
                
                # Display result
                st.success("Analysis complete!")
                st.text_area("Result:", value=result, height=300, disabled=True)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a query first.") 
