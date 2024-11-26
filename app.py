import streamlit as st
import os
from langchain.callbacks import StreamlitCallbackHandler
from rag_agent import DocumentAnalysisAgents
from langchain.agents import AgentType

st.set_page_config(page_title="RAG Agent Interface", layout="wide")

# At the very start of your app
st.write("Secrets content type:", type(st.secrets))
st.write("Secrets keys:", st.secrets.keys())

# Try this modified version for loading secrets
for key, value in dict(st.secrets).items():
    if isinstance(value, str):  # Only set if it's a string
        os.environ[key] = value

# Input area
query = st.text_area("Enter your query:", height=150)
client_id = st.text_input("Enter client ID:", value="P23")

if st.button("Run Query", type="primary"):
    if query:
        with st.spinner("Processing..."):
            try:
                callbacks = [StreamlitCallbackHandler(st.container())]
                analysis_workflow = DocumentAnalysisAgents()
                final_analysis = analysis_workflow.execute_analysis(
                    query=query,
                    meeting_notes=None,
                    client_agreements=None,
                    client_id=client_id,
                    callbacks=callbacks
                )
                st.write("### Results")
                st.write(final_analysis)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a query to continue.")
