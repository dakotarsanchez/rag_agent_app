import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents import Tool, AgentExecutor, create_react_agent, initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
import requests
import json
import litellm
from datetime import datetime, timedelta
import re
import streamlit as st
from typing import Optional

class RAGAgent:
    def __init__(self, api_key: Optional[str] = None):
        # Load environment variables and API keys
        load_dotenv()
        self.api_key = api_key or st.secrets.get("RAGIE_API_KEY") or os.getenv('RAGIE_API_KEY')
        self.openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("RAGIE_API_KEY not found in environment variables or secrets")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables or secrets")
            
        # Initialize LLM and embeddings
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=self.openai_api_key
        )
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key
        )
        
        # Initialize tools
        self.tools = [
            Tool(
                name="Meeting Notes Analysis",
                func=self._analyze_meeting_notes,
                description="Analyzes meeting notes for relevant information"
            ),
            Tool(
                name="Agreement Analysis",
                func=self._analyze_agreements,
                description="Analyzes client agreements for relevant information"
            ),
        ]
        
        # Initialize the agent with updated prompt
        prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}

{agent_scratchpad}"""

        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
        )

        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )
        
        # Initialize conversation history
        self.conversation_history = []

    def ragie_api_search(self, query: str, timeframe_info=None, client_id=None, document_type=None):
        """Execute search against Ragie API with proper filtering"""
        url = "https://api.ragie.ai/retrievals"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "query": query,
            "top_k": 5,
            "filter": {},
            "rerank": True
        }

        if client_id:
            payload["filter"]["client"] = client_id
        
        if document_type == 'meetings':
            payload["filter"]["folder"] = "test_client_meetings"
        elif document_type == 'agreements':
            payload["filter"]["folder"] = "test_client_agreements"

        response = requests.post(url, json=payload, headers=headers)
        return response.json()

    def add_to_history(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        
    def get_conversation_context(self) -> str:
        """Format conversation history into a string context."""
        if not self.conversation_history:
            return ""
            
        context = "Previous conversation:\n"
        for message in self.conversation_history:
            prefix = "User: " if message["role"] == "user" else "Assistant: "
            context += f"{prefix}{message['content']}\n"
        return context

    def process_query(self, query: str) -> str:
        """Main query processing method"""
        # Add user query to history
        self.add_to_history("user", query)
        
        # Get conversation context
        context = self.get_conversation_context()
        
        try:
            # Use the agent executor to process the query
            response = self.agent_executor.invoke(
                {
                    "input": f"{context}\n\nCurrent question: {query}"
                }
            )["output"]
            
            # Add response to history
            self.add_to_history("assistant", response)
            
            return response
            
        except Exception as e:
            error_msg = f"Error during execution: {str(e)}"
            print(error_msg)
            return error_msg

    def _analyze_meeting_notes(self, query):
        """Analyze meeting notes using Ragie API"""
        results = self.ragie_api_search(query, document_type='meetings')
        # Process the results and return insights
        return results

    def _analyze_agreements(self, query):
        """Analyze agreements using Ragie API"""
        results = self.ragie_api_search(query, document_type='agreements')
        # Process the results and return insights
        return results
