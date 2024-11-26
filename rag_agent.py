import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain.tools import Tool
import requests
import json
import litellm
from datetime import datetime, timedelta
import re

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
RAGIE_API_KEY = os.getenv('RAGIE_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_URL = os.getenv('OPENROUTER_URL')

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class DocumentAnalysisAgents:
    def __init__(self):
        # Define the LLM configuration for GPT-3.5-turbo
        llm_config = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
        }
        
        self.meeting_notes_analyst = Agent(
            role='Meeting Notes Specialist',
            goal='Extract key insights from meeting notes',
            backstory='Expert in analyzing professional meeting documentation',
            verbose=True,
            llm_config=llm_config
        )
        
        self.contract_analyst = Agent(
            role='Contract Analysis Expert',
            goal='Interpret and extract critical details from client agreements',
            backstory='Skilled in legal document interpretation and business contract analysis',
            verbose=True,
            llm_config=llm_config
        )
        
        self.synthesis_analyst = Agent(
            role='Business Intelligence Synthesizer',
            goal='Combine insights from multiple sources into a coherent narrative',
            backstory='Expert in cross-referencing and integrating complex business information',
            verbose=True,
            llm_config=llm_config
        )
        
        self.query_analyst = Agent(
            role='Query Analysis Specialist',
            goal='Identify temporal constraints and date-related requirements in user queries',
            backstory='Expert in natural language processing with focus on temporal expressions and date extraction',
            verbose=True,
            llm_config=llm_config
        )
        
        self.filter_analyst = Agent(
            role='Document Filter Analyst',
            goal='Determine appropriate document filters based on query content',
            backstory='Expert in analyzing queries to identify relevant document categories between meeting transcripts and client agreements',
            verbose=True,
            llm_config=llm_config
        )
    
    def extract_date_from_filename(self, filename):
        """Extract date from filename format P23-Till-CFO-Onboarding-review - 2024-11-18.pdf"""
        match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
        if match:
            return datetime.strptime(match.group(), '%Y-%m-%d')
        return None

    def get_date_range(self, timeframe_info):
        """Convert timeframe info into start and end dates"""
        end_date = datetime.now()
        
        if 'weeks' in timeframe_info.lower():
            weeks = int(''.join(filter(str.isdigit, timeframe_info)))
            start_date = end_date - timedelta(weeks=weeks)
        elif 'days' in timeframe_info.lower():
            days = int(''.join(filter(str.isdigit, timeframe_info)))
            start_date = end_date - timedelta(days=days)
        else:
            # Default to no date filtering
            return None, None
            
        return start_date, end_date

    def analyze_query_filters(self, query):
        filter_analysis_task = Task(
            description=f"""Analyze the following query to determine which document categories should be searched: '{query}'
                       Available categories are:
                       - test_meetings: Meeting transcripts with client P23
                       - test_client_agreements: Agreements between P23 and their clients
                       Return either 'meetings', 'agreements', or 'both' based on the query content.""",
            agent=self.filter_analyst,
            expected_output="String indicating which document categories to search"
        )
        
        crew = Crew(
            agents=[self.filter_analyst],
            tasks=[filter_analysis_task],
            verbose=True
        )
        
        return str(crew.kickoff()).strip().lower()

    def ragie_api_search(self, query, timeframe_info=None, client_id=None, document_type=None):
        url = "https://api.ragie.ai/retrievals"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {RAGIE_API_KEY}"
        }
        
        # Get date range based on timeframe
        start_date, end_date = self.get_date_range(timeframe_info) if timeframe_info else (None, None)
        
        # Determine filters based on query
        filter_type = self.analyze_query_filters(query)
        
        # Initialize base payload with correct filter structure
        payload = {
            "query": query,
            "top_k": 5,
            "filter": {},
            "rerank": True
        }

        # Add client filter if provided
        if client_id:
            payload["filter"]["client"] = client_id
        
        # Set folder filter based on document type
        if document_type == 'meetings':
            payload["filter"]["folder"] = "test_meetings"
        elif document_type == 'agreements':
            payload["filter"]["folder"] = "test_client_agreements"

        # Print API call details for debugging
        print("\n=== API Call Details ===")
        print(f"URL: {url}")
        print(f"Headers: {json.dumps(headers, indent=2)}")
        print(f"Payload: {json.dumps(payload, indent=2)}")

        try:
            response = requests.post(url, json=payload, headers=headers)
            print(f"Response Status Code: {response.status_code}")
            print(f"Response Body: {json.dumps(response.json(), indent=2)}\n")
            
            chunks = response.json()['scored_chunks']
            
            # Filter and sort chunks by date if timeframe is specified
            if start_date and end_date:
                filtered_chunks = []
                for chunk in chunks:
                    chunk_date = self.extract_date_from_filename(chunk.get('source', ''))
                    if chunk_date and start_date <= chunk_date <= end_date:
                        filtered_chunks.append(chunk)
                
                # Sort by date (most recent first) and adjust scores to favor recent documents
                filtered_chunks.sort(key=lambda x: self.extract_date_from_filename(x.get('source', '')), reverse=True)
                for i, chunk in enumerate(filtered_chunks):
                    # Boost score for more recent documents
                    chunk['score'] = chunk['score'] * (1 + (len(filtered_chunks) - i) * 0.1)
                
                return filtered_chunks
            
            return chunks
        except Exception as e:
            return f"Error in API search: {str(e)}"
    
    def create_tasks(self, query, meeting_notes, client_agreements):
        meeting_notes_task = Task(
            description=f"Analyze meeting notes to extract insights related to: {query}. Notes: {meeting_notes}",
            agent=self.meeting_notes_analyst,
            expected_output="Structured summary of key meeting insights"
        )
        
        contract_analysis_task = Task(
            description=f"Analyze client agreements to extract formal details related to: {query}. Agreements: {client_agreements}",
            agent=self.contract_analyst,
            expected_output="Detailed contract insights and legal interpretations"
        )
        
        synthesis_task = Task(
            description="Synthesize insights from meeting notes and client agreements",
            agent=self.synthesis_analyst,
            expected_output="Comprehensive, coherent response integrating all available information"
        )
        
        return [meeting_notes_task, contract_analysis_task, synthesis_task]
    
    def analyze_query_timeframe(self, query):
        query_analysis_task = Task(
            description=f"Analyze the following query for any date-related constraints or timeframes: '{query}'. "
                       f"If found, specify the exact timeframe (e.g., 'last 7 days', 'past month', specific date). "
                       f"If no timeframe is mentioned, return 'no_timeframe'.",
            agent=self.query_analyst,
            expected_output="Structured timeframe information or 'no_timeframe'"
        )
        
        crew = Crew(
            agents=[self.query_analyst],
            tasks=[query_analysis_task],
            verbose=True
        )
        
        return crew.kickoff()
    
    def execute_analysis(self, query, meeting_notes, client_agreements, client_id):
        # Get timeframe analysis
        timeframe_output = self.analyze_query_timeframe(query)
        timeframe = str(timeframe_output).strip()
        
        # Separate API calls for meetings and agreements
        filter_type = self.analyze_query_filters(query)
        
        retrieved_meetings = []
        retrieved_agreements = []
        
        if filter_type in ['meetings', 'both']:
            retrieved_meetings = self.ragie_api_search(
                query, 
                timeframe, 
                client_id,
                document_type='meetings'
            )
            
        if filter_type in ['agreements', 'both']:
            retrieved_agreements = self.ragie_api_search(
                query, 
                timeframe, 
                client_id,
                document_type='agreements'
            )
        
        tasks = self.create_tasks(query, retrieved_meetings, retrieved_agreements)
        
        crew = Crew(
            agents=[
                self.meeting_notes_analyst,
                self.contract_analyst,
                self.synthesis_analyst,
                self.query_analyst
            ],
            tasks=tasks,
            verbose=True
        )
        
        result = crew.kickoff()
        return result

def main():
    query = "What is David's relationship with Sedulen like?"
    client_id = "P23"  # or however you want to identify the client
    
    meeting_notes = "Sample meeting notes content about P23's contracts and business agreements"
    client_agreements = "Sample client agreements content detailing contract terms for P23"
    
    analysis_workflow = DocumentAnalysisAgents()
    final_analysis = analysis_workflow.execute_analysis(
        query, 
        meeting_notes, 
        client_agreements,
        client_id=client_id
    )
    
    print("Final Analysis:", final_analysis)

if __name__ == "__main__":
    main()
