import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.agents import Tool, AgentExecutor, create_react_agent, initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
import requests
import json
import litellm
from datetime import datetime, timedelta
import re
import streamlit as st
from typing import Optional, List, Dict
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from crewai import Agent, Task, Crew, Process
from pydantic import BaseModel, Field

class TemporalQuery(BaseModel):
    query_type: str = Field(description="Type of temporal query (e.g., 'latest_n' or 'date_range')")
    count: int = Field(description="Number of meetings to retrieve when query_type is 'latest_n'")
    start_date: str = Field(description="Start date for date range queries (YYYY-MM-DD)")
    end_date: str = Field(description="End date for date range queries (YYYY-MM-DD)")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query_type": "latest_n",
                    "count": 5,
                    "start_date": "",
                    "end_date": ""
                }
            ]
        }
    }

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
        
        # Initialize the base tools
        self.tools = self._initialize_tools()
        
        # Initialize CrewAI agents
        self.meeting_analyst = Agent(
            role='Meeting Notes Analyst',
            goal='Analyze meeting notes to extract relevant information and insights',
            backstory='Expert at analyzing meeting notes and extracting key information',
            tools=[self.tools[0]],  # Meeting Notes Analysis tool
            llm=self.llm,
            verbose=True
        )
        
        self.agreement_analyst = Agent(
            role='Agreement Analyst',
            goal='Analyze agreements to extract relevant information and requirements',
            backstory='Expert at analyzing legal agreements and contractual documents',
            tools=[self.tools[1]],  # Agreement Analysis tool
            llm=self.llm,
            verbose=True
        )
        
        # Initialize the crew
        self.crew = Crew(
            agents=[self.meeting_analyst, self.agreement_analyst],
            tasks=[],
            process=Process.sequential
        )
        
        # Initialize conversation history
        self.conversation_history = []

        # Add temporal query parser
        self.temporal_parser = PydanticOutputParser(pydantic_object=TemporalQuery)
        self.temporal_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at understanding temporal requirements in queries about meetings.
            Current date: {current_date}
            Available meeting dates: {available_dates}
            
            Analyze the query to determine what meetings are relevant based on time requirements."""),
            ("user", "{query}"),
            ("system", "Provide your analysis in the following format:\n{format_instructions}")
        ])

    def _initialize_tools(self):
        """Initialize the tools for the agents"""
        return [
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

    def get_ragie_documents(self, folder_name: str) -> List[Dict]:
        """Get all documents from a Ragie folder"""
        url = f"https://api.ragie.ai/documents"
        params = {
            "page_size": 99,
            "filter": {"folder": {"$eq": folder_name}}
        }
        
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json().get('documents', [])
        except Exception as e:
            print(f"Error fetching documents: {e}")
            return []

    def analyze_temporal_requirements(self, query: str, available_dates: List[str]) -> TemporalQuery:
        """Use LLM to analyze temporal requirements in query"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        formatted_prompt = self.temporal_prompt.format_messages(
            query=query,
            current_date=current_date,
            available_dates=", ".join(available_dates),
            format_instructions=self.temporal_parser.get_format_instructions()
        )
        
        response = self.llm.invoke(formatted_prompt)
        return self.temporal_parser.parse(response.content)

    def filter_meetings_by_date(self, meetings: List[Dict], temporal_info: TemporalQuery) -> List[Dict]:
        """Filter meetings based on temporal requirements"""
        # Extract and sort dates
        dated_meetings = []
        for meeting in meetings:
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', meeting['name'])
            if date_match:
                meeting_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                dated_meetings.append((meeting_date, meeting))
        
        dated_meetings.sort(key=lambda x: x[0], reverse=True)
        
        if temporal_info.query_type == "latest_n":
            return [meeting for _, meeting in dated_meetings[:temporal_info.count]]
        elif temporal_info.query_type == "date_range":
            start_date = datetime.strptime(temporal_info.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(temporal_info.end_date, "%Y-%m-%d")
            return [
                meeting for date, meeting in dated_meetings
                if start_date <= date <= end_date
            ]
        
        return [meeting for _, meeting in dated_meetings]

    def ragie_api_search(self, query: str, timeframe_info=None, client_id=None, document_type=None):
        """Execute search against Ragie API with proper filtering"""
        # If it's a meeting query, first get relevant meetings
        if document_type == 'meetings':
            # Get all available meetings
            all_meetings = self.get_ragie_documents("test_client_meetings")
            
            # Extract available dates for context
            available_dates = []
            for meeting in all_meetings:
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', meeting['name'])
                if date_match:
                    available_dates.append(date_match.group(1))
            
            # Analyze temporal requirements
            temporal_info = self.analyze_temporal_requirements(query, available_dates)
            
            # Filter meetings based on temporal requirements
            relevant_meetings = self.filter_meetings_by_date(all_meetings, temporal_info)
            
            # Get content for relevant meetings
            document_ids = [meeting['id'] for meeting in relevant_meetings]
            if document_ids:
                payload = {
                    "query": query,
                    "top_k": 5,
                    "filter": {
                        "folder": "test_client_meetings",
                        "document_id": {"$in": document_ids}
                    },
                    "rerank": True
                }
            else:
                payload = {
                    "query": query,
                    "top_k": 5,
                    "filter": {
                        "folder": "test_client_meetings"
                    },
                    "rerank": True
                }
        else:
            # Original agreement search logic
            payload = {
                "query": query,
                "top_k": 5,
                "filter": {
                    "folder": document_type
                },
                "rerank": True
            }

        # Execute API call
        url = "https://api.ragie.ai/retrievals"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }
        
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
        """Main query processing method using CrewAI"""
        # Add user query to history
        self.add_to_history("user", query)
        
        try:
            # Create tasks based on the query
            tasks = [
                Task(
                    description=f"Analyze the following query and provide relevant information: {query}",
                    agent=self.meeting_analyst
                ),
                Task(
                    description=f"Review agreements for information related to: {query}",
                    agent=self.agreement_analyst
                )
            ]
            
            # Update crew tasks
            self.crew.tasks = tasks
            
            # Execute the crew
            result = self.crew.kickoff()
            
            # Add response to history
            self.add_to_history("assistant", result)
            
            return result
            
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

    def get_recent_meeting_summaries(self, num_meetings: int = 3):
        """Get summaries for the most recent meetings"""
        # Get documents from test_meetings folder
        url = f"https://api.ragie.ai/documents?page_size={num_meetings}&filter=%7B%22folder%22%3A%20%7B%22%24eq%22%3A%20%22test_meetings%22%7D%7D"
        
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }
        
        try:
            print("Fetching documents...")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            documents = response.json().get('documents', [])
            meeting_summaries = []
            
            # For each document, fetch its summary
            for doc in documents:
                try:
                    doc_id = doc['id']
                    summary_url = f"https://api.ragie.ai/documents/{doc_id}/summary"
                    summary_response = requests.get(summary_url, headers=headers)
                    summary_response.raise_for_status()
                    
                    meeting_summaries.append({
                        'name': doc.get('name', 'Unnamed Meeting').replace('.pdf', ''),
                        'summary': summary_response.json().get('summary', 'No summary available')
                    })
                    
                except Exception as e:
                    print(f"Error fetching summary for document {doc.get('id')}: {e}")
                    
            return meeting_summaries
            
        except Exception as e:
            print(f"Error fetching meetings: {e}")
            return []
