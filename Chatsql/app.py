import streamlit as st
import os
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
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
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from sqlalchemy import create_engine
import psycopg2
from dotenv import load_dotenv
load_dotenv()

postgres_conector_uri = os.getenv("postgres_connector_uri")

st.set_page_config(page_title="🦜🔗 Groq SQL Agent with PostgreSQL Db")
st.title("🦜🔗 Groq SQL Agent with PostgreSQL Db")

LOCALDB = "USE_LOCALDB"
SQL = "USE_SQL"

radio_option = ["Use postgresql db - student_model", "Connect to your own db"]

selected_opt =st.sidebar.radio(label="Choose an option", options = radio_option)

if selected_opt == radio_option[1]:
    postgres_uri = SQL
    postgres_host = st.sidebar.text_input("Enter your postgres host")
    postgres_user = st.sidebar.text_input("Enter your postgres user")
    postgres_password = st.sidebar.text_input("Enter your postgres password", type = "password")
    postgres_db = st.sidebar.text_input("Enter your postgres db name")
    postgres_port = st.sidebar.text_input("Enter your postgres port", value = "5432")
    # postgres_uri = f"postgresql+psycopg2://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

else:
    postgres_uri = LOCALDB

if not postgres_uri:
    st.warning("Please enter all the details to connect to your postgres db")

api_key = st.sidebar.text_input("Enter your groq api_key:", type="password")

if not api_key:
    st.warning("Please enter your groq api key to connect to the llm")

# LLM model
llm = ChatGroq(model_name = "Llama3-8b-8192",groq_api_key = api_key, streaming=True)

@st.cache_resource(ttl = "2h")
def configure_db(postgres_uri,postgres_host,postgres_user,postgres_password,postgres_db,postgres_port):
    if postgres_uri == SQL:
        if not (postgres_host and postgres_user and postgres_password and postgres_db and postgres_port):
            st.warning("Please enter all the details to connect to your postgres db")
            st.stop()
        postgres_uri = f"postgresql+psycopg2://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
        return SQLDatabase(create_engine(postgres_uri))
    elif postgres_uri == LOCALDB:
        # local db connection
        return SQLDatabase(create_engine(postgres_conector_uri), include_tables=["students"])
    
# call the function configure_db
if postgres_uri == SQL:
    # call the function
    db = configure_db(postgres_uri,postgres_host,postgres_user,postgres_password,postgres_db,postgres_port)
elif postgres_uri == LOCALDB:
    # call the function
    db = configure_db(postgres_uri = LOCALDB,postgres_host = "localhost",postgres_user = "postgres",postgres_password = "12345678",postgres_db = "student_model",postgres_port = 5432)

# create toolkit

toolkit = SQLDatabaseToolkit(db = db, llm = llm)

agent = create_sql_agent(llm = llm, toolkit = toolkit,verbose = True,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role" : "assistant", "content": "How can I help you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input(placeholder="Ask anything from the database")

if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        # to display the thoughts and actions using the streamlit callback handler
        streamlit_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts = True)
        response = agent.run(prompt, callbacks=[streamlit_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)