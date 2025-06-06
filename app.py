import streamlit as st
import pandas as pd
import os
import time
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.memory.buffer import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from llm import run_agent

st.set_page_config(page_title="ArXiv Agent")

st.title("💻 ArXiv Agent")

tone = st.selectbox(label="Select the tone", options=('Literature Survey', 'News', 'Anime', 'Star wars', 'GenZ'))

##############
# SIDEBAR    #
##############

# Sidebar selection for API
connection_method = st.sidebar.selectbox("Model Options", ("Open AI", "Ollama"))

# Ollama configuration
if connection_method == "Ollama":
    if 'ollama' not in st.session_state:
        st.session_state.ollama = {}
        st.session_state.ollama['model_id'] = st.sidebar.text_input("Model ID")
    else:
        st.session_state.ollama['model_id'] = st.sidebar.text_input("Model ID", value=st.session_state.ollama['model_id'])

    # Clear OpenAI state
    if 'openai' in st.session_state:
        del st.session_state.openai

# OpenAI configuration
else:
    if 'openai' not in st.session_state:
        st.session_state.openai = {}
        st.session_state.openai['openai_api_key'] = st.sidebar.text_input("OpenAI API key", type="password")
        st.session_state.openai['openai_model'] = st.sidebar.text_input("OpenAI Model", value="o4-mini")
    else:
        st.session_state.openai['openai_api_key'] = st.sidebar.text_input("OpenAI API key", type="password", value=st.session_state.openai['openai_api_key'])
        st.session_state.openai['openai_model'] = st.sidebar.text_input("OpenAI Model", value=st.session_state.openai['openai_model'])

    # Clear Ollama state
    if 'ollama' in st.session_state:
        del st.session_state.ollama


if 'tavily_api' not in st.session_state:
    st.session_state.tavily_api = st.sidebar.text_input("Tavily API", type="password")
else:
    st.session_state.tavily_api = st.sidebar.text_input("Tavily API", type="password", value=st.session_state['tavily_api'])

# print(st.session_state.get("ollama", None))
# print(st.session_state.get('openai', None))

if 'openai' in st.session_state:
    model_info = {
        'type': 'openai',
        'api_key': st.session_state['openai']['openai_api_key'],
        'model_id': st.session_state['openai']['openai_model']
    }
else:
    model_info = {
        'type': 'ollama',
        'model_id': st.session_state['ollama']['model_id']
    }

############
# MEMORY   #
############
# initial_message = """
#     You are a research assistant with access to the arXiv paper search and HTML retrieval tools. Your job is to answer user queries by sourcing content directly from academic research papers.

# Use the following rules:

# 1. Search arXiv for papers that are:
#    - Relevant to the user query using keywords and semantic expansion.
#    - Sorted by both **relevance** and **publication date** (present both top and recent results).

# 2. Retrieve paper metadata (title, authors, date, link) and full content (via HTML or PDF parsing) for top matches.

# 3. Extract facts, equations, results, and insights **only from retrieved papers**. Do not make up or hallucinate any information.

# 4. Quote or cite the exact sentence/paragraph from the paper with its reference [title, arXiv link].

# 7. Provide critic, advantages, disadvantages and gaps in the paper.

# 8. Access the impact of papers based on web search, collecting information on citation count.

# 9. Always add references and link at the end.

# 10. Use Tavily search tool if arxiv does not have the answers. DO NOT CREATE FAKE LINKS.
# ---

# ### Prompt Input Format
# Query: <query>
# ---
# """ 
if tone == "News":
    initial_message = """
    You are the editor at CNN news. To provide crisp facts.
    Provide highlight about the paper after doing research using paper search, summarizing and paringly using the internet.
    Provide facts after facts like breaking news.
    Pick the latest recent articles.
    DO NOT MAKE UP DATA.
    """
elif tone == "Anime":
    initial_message = """
    You are an anime geek. You provide facts from the research like anime facts.
    Provide deticated response based on the paper search, summarizing and paringly using the internet.
    Provide funny and goofy references in anime style.
    DO NOT MAKE UP DATA.
    """
elif tone == "GenZ":
    initial_message = """
    You are a GenZ teenager. You use slang words all the time.
    Provide legit information about the query based on paper search, summarizing and paringly using the internet.
    The response should be true but annoying.
    DO NOT MAKE UP DATA. Provide references in GenZ grammar.
    """
elif tone == "Literature Survey":
    initial_message = """
    You are a PhD graduate, provide detailed literature review section based on prompt.
    Proper a section that can be added and submitted in a reputed journal.
    The reponse should be connected and maintain flow, highlighting gaps.
    DO NOT MAKE UP DATA. Provide academic citations wherever possible.
    """
elif tone == "Star wars":
    initial_message = """
    You are an Darth Vader. You are loyal to the Sith lord and don't lie.
    DO NOT MAKE UP DATA. I am the Sith lord, give me information.
    Provide only citations if you know it matches to satisfy the Sith lord. Do not give wrong links.
    Include references from Star wars.
    """

if 'tone' not in st.session_state:
    st.session_state['tone'] = tone
else:
    st.session_state['tone'] = tone
    del st.session_state.memory


if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    # Create the prompt object
    prompt = ChatPromptTemplate.from_template(initial_message)

    # Format the prompt with actual inputs
    formatted_prompt = prompt
    st.session_state.memory.chat_memory.add_message(SystemMessage(content=initial_message))


###########
# BODY    #
###########


if 'memory' in st.session_state:
    for message in st.session_state.memory.chat_memory.messages:
        role = message.type  # 'human' or 'ai' or 'system'
        content = message.content
        if role in ['human', 'ai']:
            with st.chat_message(role):
                st.markdown(content)

prompt = st.chat_input("Ask me about a computer science research topic!")
if prompt:
    try:
    # if (('openai' in st.session_state and st.session_state.openai['openai_api_key'] != "") or ('ollama' in st.session_state and st.session_state.ollama['model_id'] != "")) and ('tavily_api' in st.session_state and st.session_state.tavily_api != ''):
        with st.chat_message("user"):
            st.markdown(prompt)
        # st.session_state.memory.chat_memory.add_message(HumanMessage(content=prompt))
        with st.chat_message("assistant"):
            ai_res = run_agent(model_info=model_info, memory=st.session_state.memory, user_query=prompt, tavily_api_key=st.session_state.tavily_api)
            st.markdown(ai_res)
        # st.session_state.memory.chat_memory.add_message(AIMessage(content=ai_res))
    except Exception as e:
        st.warning("Provide proper details in the sidebar else model failed because of limitations")
        st.error(f"Error details: {e}")
