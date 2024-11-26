import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify secrets are loaded
if not st.secrets:
    raise ValueError("Streamlit secrets not found")

# Add debug logging
st.write("Debug: Starting application")

# Initialize your RAG agent with the secrets
from rag_agent import RAGAgent
st.write("Debug: Importing RAGAgent")

try:
    agent = RAGAgent()
    st.write("Debug: RAGAgent initialized successfully")
except Exception as e:
    st.error(f"Debug: Failed to initialize RAGAgent: {str(e)}")
    st.stop()

# Initialize the session state for the agent if it doesn't exist
if 'agent' not in st.session_state:
    try:
        st.session_state.agent = agent
    except Exception as e:
        st.error(f"Error initializing RAGAgent: {str(e)}")
        st.stop()

# Add Recent Meeting Summaries section
st.title("Recent Meeting Summaries")

try:
    recent_summaries = st.session_state.agent.get_recent_meeting_summaries(3)
    
    if not recent_summaries:
        st.warning("No recent meetings found or error fetching meetings.")
    else:
        for meeting in recent_summaries:
            with st.expander(f"{meeting['name']}"):
                st.write(meeting['summary'])
except Exception as e:
    st.error(f"Error fetching meeting summaries: {str(e)}")

# Add a separator
st.markdown("---")

# Chat Interface
st.title("Document Analysis Chat")

# Get user input
user_input = st.text_input("Ask a question about your documents:")

if user_input:
    st.write("Debug: Processing query...")
    try:
        response = st.session_state.agent.process_query(user_input)
        st.write("Debug: Query processed successfully")
        st.write(response)
    except Exception as e:
        st.error(f"Debug: Error processing query: {str(e)}")

# Optional: Display conversation history
if st.session_state.agent.conversation_history:
    st.subheader("Conversation History")
    for message in st.session_state.agent.conversation_history:
        prefix = "You: " if message["role"] == "user" else "Assistant: "
        st.text(f"{prefix}{message['content']}")
