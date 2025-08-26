import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun,DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader,PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents.initialize import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from dotenv import load_dotenv
load_dotenv()

# Don't need to define the api_key

# Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results = 1,doc_content_chars_max = 200)
arxiv = ArxivQueryRun(api_wrapper = arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results = 1, doc_content_chars_max = 200)
wiki = WikipediaQueryRun(api_wrapper = wiki_wrapper)

search = DuckDuckGoSearchRun(name = "Search")

st.title("🦜🔗 Groq Search Engine with Agents")

st.sidebar.title('Settings')
api_key = st.sidebar.text_input("Enter your groq api key:", type="password")

if "messages" not in st.session_state:
         st.session_state["messages"] = [
            {"role" : "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
         ]

for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input(placeholder= "What is machine learning?")
if prompt:
        st.session_state.messages.append({"role":"user", "content": prompt})
        st.chat_message("user").write(prompt)

        llm = ChatGroq(model_name = "Llama3-8b-8192", groq_api_key=api_key,streaming=True)
        tools = [arxiv, wiki, search]
        search_agent = initialize_agent(tools,llm,agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose = True,handling_passing_errors = True)
        with st.chat_message("assistant"):
            streamlit_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts = True)
            response = search_agent.run(st.session_state.messages,callbacks = [streamlit_cb])
            st.session_state.messages.append({"role":"assistant", "content" : response})
            st.write(response)