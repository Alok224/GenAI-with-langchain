import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
import groq
# api key
# groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['OLLAMA_API_KEY'] = os.getenv('OLLAMA_API_KEY')
os.environ["USER_AGENT"] = "GenAI-with-langchain/1.0"

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model= "mxbai-embed-large")
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistaint. Please response to the user's queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question,api_key,llm,temperature, max_tokens):
    groq_api_key = api_key
    llm = ChatGroq(model_name = "gemma2-9b-It",groq_api_key = groq_api_key)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke(
        {'question' : question}
    )
    return answer


# title of the app

import streamlit as st

st.title("Enhanced Q&A chatbot with groq")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")


# # dropdown to select various groq models

llm = st.sidebar.selectbox("select an groq model", ["gemma2-9b-It", "gemma2-9b-It-8k", "gemma2-9b-It-32k"])

temperature = st.sidebar.slider("Select temperature", min_value= 0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Select max tokens", min_value= 50, max_value=300, value=150)

# main interface for user input

st.write("Ask a question to the chatbot:")

user_input = st.text_input("Your question:")

if user_input:
    with st.spinner("Generating response..."):
        try:
            response = generate_response(user_input, api_key, llm, temperature, max_tokens)
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.write("Please enter a question to get started.")