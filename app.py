import streamlit as st
from rag_agent import RAGAgent

# Initialize the session state for the agent if it doesn't exist
if 'agent' not in st.session_state:
    st.session_state.agent = RAGAgent()

st.title("Document Analysis Chat")

# Get user input
user_input = st.text_input("Ask a question about your documents:")

if user_input:
    # Process the query using the RAG agent
    response = st.session_state.agent.process_query(user_input)
    st.write(response)

# Optional: Display conversation history
if st.session_state.agent.conversation_history:
    st.subheader("Conversation History")
    for message in st.session_state.agent.conversation_history:
        prefix = "You: " if message["role"] == "user" else "Assistant: "
        st.text(f"{prefix}{message['content']}")
