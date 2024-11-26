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

class DocumentAnalysisAgents:
    def __init__(self):
        # Load API keys in the constructor
        self.RAGIE_API_KEY = os.getenv('RAGIE_API_KEY')
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
        self.OPENROUTER_URL = os.getenv('OPENROUTER_URL')
        
        # Define the LLM configuration for GPT-3.5-turbo
        llm_config = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
        }
        
        self.llm = ChatOpenAI(temperature=0)
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize your tools and agents here
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
            # Add your other tools here
        ]
        
        # Update the prompt template to include all required variables
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

        # Create the agent with the updated prompt
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

    def _analyze_meeting_notes(self, query):
        # Your meeting notes analysis logic here
        pass

    def _analyze_agreements(self, query):
        # Your agreement analysis logic here
        pass

    def execute_analysis(self, query, meeting_notes, client_agreements, client_id, callbacks=None):
        # Use the agent_executor instead of creating a new agent
        try:
            return self.agent_executor.run(
                input=query,
                callbacks=callbacks
            )
        except Exception as e:
            print(f"Error during execution: {str(e)}")
            raise
