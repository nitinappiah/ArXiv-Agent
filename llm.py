from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.memory.buffer import ConversationBufferMemory
import pandas as pd
from langchain.agents import (
    AgentExecutor, create_react_agent, create_structured_chat_agent, create_tool_calling_agent
)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
import re
import os
from langchain_core.tools.simple import Tool
import requests
import xmltodict
import json
from rag import create_vector_store, retrieve_documents

def get_papers(search_query, sort_by, model_info):
    """
        sort_by - relevance, lastUpdatedDate, submittedDate
    """
    # Make HTTP request to get XML    
    max_results = 10
    url = f"https://export.arxiv.org/api/query?search_query=all:{search_query}&start=0&max_results={max_results}&sort_by={sort_by}"
    response = requests.get(url)

    # Parse XML to Python dictionary
    xml_data = response.text
    parsed_dict = xmltodict.parse(xml_data)

    entries = parsed_dict.get("feed", {}).get("entry", "No relevant paper available")
    db = create_vector_store(entries, model_info)
    retrieved_documents = retrieve_documents(db, search_query, model_info)
    return str(retrieved_documents)

def get_paper_html(abs_link):
    html_link = abs_link.replace("/abs/", "/html/")
    response = requests.get(html_link)
    print(response.text)

def get_recent_papers(search_query, model_info):
    return get_papers(search_query=search_query, sort_by="submittedDate", model_info=model_info)

def get_relevant_papers(search_query, model_info):
    return get_papers(search_query=search_query, sort_by="relevance", model_info=model_info)


def run_agent(model_info, memory, tavily_api_key, user_query):
    tools = [
        Tool(
            name="PaperSearchRecent",
            func=lambda q: str(get_recent_papers(search_query=q, model_info=model_info)),
            description="Get a list of recent papers published on the topic sorted by submitted date"
        ),
        Tool(
            name="PaperSearchRelevant",
            func=lambda q: str(get_relevant_papers(search_query=q, model_info=model_info)),
            description="Get a list of relevant papers to the topic based on key word match"
        ),
        Tool(
            name="PaperHTMLExtractor",
            func=get_paper_html,
            description="Get the paper content given the abs_link"
        ),
        TavilySearchResults(tavily_api_key=tavily_api_key)
    ]

    prompt = hub.pull("hwchase17/openai-tools-agent")

    # select llm model
    llm = None
    if model_info['type'] == "openai":
        llm = ChatOpenAI(model=model_info['model_id'], openai_api_key=model_info['api_key'])
    elif model_info['type'] == "ollama":
        llm = ChatOllama(model=model_info['model_id'])
    
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True
    )

    # import pdb; pdb.set_trace()
    result = agent_executor.invoke({"input": user_query})
    return result['output']