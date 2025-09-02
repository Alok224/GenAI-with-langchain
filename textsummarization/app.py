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
import validators
from langchain.schema import AIMessage,SystemMessage,HumanMessage
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# api_key = os.getenv("GROQ_API_KEY")

# give the title name

st.set_page_config(page_title="Langchain: Summarize Text From YT Or Website")
st.title("Langchain: Summarize Text From YT Or Website")

st.subheader("Summarize URL")

with st.sidebar:
    groq_api_key = st.sidebar.text_input(label="Your GroqAPI Key", type= "password")

generic_url = st.text_input("URL",label_visibility="collapsed")


final_generic_template = """ Provide the final summary of the entire content with these important points:
Add a entire summary head, Start the precise summary with an introduction and provide the summary in number
points for the content
content:{text}
"""

final_prompt = PromptTemplate(input_variables=["text"],template=final_generic_template)

generic_template = """ Provide the concise summary of the following content in 300 words.

content: {text}
"""
chunk_prompt = PromptTemplate(input_variables=["text"],template=generic_template)

if st.button("Summarize the content from YT or Website"):
    # validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.warning("Please fill the required fields")
    elif not validators.url(generic_url):
        st.warning("Please Provide a validate url")

    else:
        try:
            with st.spinner("Please wait..."):
                llm = ChatGroq(model_name = "llama-3.1-8b-instant", groq_api_key = groq_api_key)
                # loading the website or youtube data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info = True)
                # If generic url is a website url 
                else:
                    headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                            "Accept-Language": "en-US,en;q=0.9"
                        }
                    loader = UnstructuredURLLoader(urls=[generic_url],ssl_verify = False, headers = headers)
                docs = loader.load()
                split_docs = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 200)
                final_docs = split_docs.split_documents(docs)

                # chain summarization
                chain = load_summarize_chain(llm = llm,chain_type = "map_reduce",verbose=True,map_prompt = chunk_prompt, combine_prompt = final_prompt)
                output_summary = chain.run(final_docs)

                st.success(output_summary)

        except Exception as e:
            st.exception(e)
