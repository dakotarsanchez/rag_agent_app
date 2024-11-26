import streamlit as st
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Verify secrets are loaded
if not st.secrets:
    raise ValueError("Streamlit secrets not found")

# Initialize your RAG agent with the secrets
from rag_agent import RAGAgent
logger.debug("Importing RAGAgent")

try:
    agent = RAGAgent()
    logger.debug("RAGAgent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAGAgent: {str(e)}", exc_info=True)
    st.error(f"Failed to initialize RAGAgent: {str(e)}")
    st.stop()

# Initialize the session state for the agent if it doesn't exist
if 'agent' not in st.session_state:
    try:
        st.session_state.agent = agent
        logger.debug("Agent added to session state")
    except Exception as e:
        logger.error(f"Error initializing session state: {str(e)}", exc_info=True)
        st.error(f"Error initializing session state: {str(e)}")
        st.stop()

# Add Recent Meeting Summaries section
st.title("Recent Meeting Summaries")

try:
    logger.debug("Fetching recent meeting summaries")
    recent_summaries = st.session_state.agent.get_recent_meeting_summaries(3)
    
    if not recent_summaries:
        logger.warning("No recent meetings found")
        st.warning("No recent meetings found.")
    else:
        logger.debug(f"Found {len(recent_summaries)} recent meetings")
        for meeting in recent_summaries:
            with st.expander(f"{meeting['name']}"):
                st.write(meeting['summary'])
except Exception as e:
    logger.error(f"Error fetching meeting summaries: {str(e)}", exc_info=True)
    st.error(f"Error fetching meeting summaries: {str(e)}")

# Chat Interface
st.title("Document Analysis Chat")

# Get user input
user_input = st.text_input("Ask a question about your documents:")

if user_input:
    logger.debug(f"Processing query: {user_input}")
    try:
        response = st.session_state.agent.process_query(user_input)
        logger.debug("Query processed successfully")
        st.write(response)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        st.error(f"Error processing query: {str(e)}")
        st.error(f"Debug: Error processing query: {str(e)}")

# Optional: Display conversation history
if st.session_state.agent.conversation_history:
    st.subheader("Conversation History")
    for message in st.session_state.agent.conversation_history:
        prefix = "You: " if message["role"] == "user" else "Assistant: "
        st.text(f"{prefix}{message['content']}")
