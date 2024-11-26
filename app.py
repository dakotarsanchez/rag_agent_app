import streamlit as st
import os
from rag_agent import DocumentAnalysisAgents
from langchain.agents import AgentType
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="RAG Agent Interface", layout="wide")

# Add proper error handling for API key
try:
    OPENAI_API_KEY = st.text_input("Enter your OpenAI API Key:", type="password")
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = str(OPENAI_API_KEY)
        
        # Optional: Add other API keys if needed
        RAGIE_API_KEY = st.text_input("Enter your RAGIE API Key:", type="password")
        if RAGIE_API_KEY:
            os.environ["RAGIE_API_KEY"] = str(RAGIE_API_KEY)
    else:
        st.warning("Please enter your OpenAI API key to continue.")
except Exception as e:
    st.error(f"Error setting API key: {str(e)}")

# Input area
query = st.text_area("Enter your query:", height=150)
client_id = st.text_input("Enter client ID:", value="P23")

if st.button("Run Query", type="primary"):
    if query and os.getenv("OPENAI_API_KEY"):
        with st.spinner("Processing..."):
            try:
                # Create callback handler for Streamlit
                callbacks = [StreamlitCallbackHandler(st.container())]
                
                analysis_workflow = DocumentAnalysisAgents()
                final_analysis = analysis_workflow.execute_analysis(
                    query=query,
                    meeting_notes=None,
                    client_agreements=None,
                    client_id=client_id,
                    callbacks=callbacks  # Pass callbacks to the agent
                )
                st.write("### Results")
                st.write(final_analysis)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                # Add more detailed error logging
                st.error("Please check that your DocumentAnalysisAgents class is properly configured with tools")
    else:
        st.warning("Please enter both a query and API key to continue.")
