import streamlit as st
from rag_agent import RAGAgent

# Initialize the session state for the agent if it doesn't exist
if 'agent' not in st.session_state:
    try:
        st.session_state.agent = RAGAgent()
    except Exception as e:
        st.error(f"Error initializing RAGAgent: {str(e)}")
        st.stop()

# Add Recent Meeting Summaries section
st.title("Recent Meeting Summaries")

try:
    recent_summaries = st.session_state.agent.get_recent_meeting_summaries(3)
    
    if not recent_summaries:
        st.warning("No recent meetings found.")
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
    # Process the query using the RAG agent
    response = st.session_state.agent.process_query(user_input)
    st.write(response)

# Optional: Display conversation history
if st.session_state.agent.conversation_history:
    st.subheader("Conversation History")
    for message in st.session_state.agent.conversation_history:
        prefix = "You: " if message["role"] == "user" else "Assistant: "
        st.text(f"{prefix}{message['content']}")
