import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question's asked"),
        ("user", "Question: {question}")
    ]
)

# Streamlit framework

st.title("langchain demo with goggle gemma model")
input_text = st.text_input("Enter your question here")

# Ollama gemma 2b model
llm = ChatOllama(model = "gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    with st.spinner("Generating response..."):
        try:
            response = chain.invoke({"question": input_text})
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")