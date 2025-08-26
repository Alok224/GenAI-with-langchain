import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain


from dotenv import load_dotenv
load_dotenv()

# load environment variables
os.environ['OLLAMA_API_KEY'] = os.getenv('OLLAMA_API_KEY')
os.environ["USER_AGENT"] = "GenAI-with-langchain/1.0"
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['HUGGINGFACE_API_KEY'] = os.getenv('HUGGINGFACE_API_KEY')

# you can also get your api key
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="gemma2-9b-It", groq_api_key=groq_api_key)



prompt = ChatPromptTemplate.from_template(
    # give the context

    """
    Answer the question based on the context provided. 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    please provide the most accurate answer based on the context provided.
    <context>
    {context}
    </context>
    Question: {question}
    """
)

def create_vector_embeddings():
    if os.path.exists("chroma_store"):  
        # Load existing store
        st.session_state.vectors = Chroma(
            persist_directory="chroma_store",
            embedding_function=st.session_state.embeddings
        )
    else:
        # Ingest new documents
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name = "all-mpnet-base-v2")
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.documents = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        st.session_state.split_documents = st.session_state.text_splitter.split_documents(
            st.session_state.documents
        )

        #  Create and persist vector store
        st.session_state.vectors = Chroma.from_documents(
            st.session_state.split_documents,
            embedding=st.session_state.embeddings,
            persist_directory="chroma_store"
        )
        
st.title("RAG Document Q&A with groq and ollama")
        
user_prompt = st.text_input("Enter your question from research papers:")

if st.button("Generate Answer"):
    create_vector_embeddings()
    st.write("Your vector database is ready")

import time

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Generate Answer' first to build the vector database.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'question': user_prompt})
        print(f"Response time: {time.process_time() - start}")

        st.write(response['answer'])

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('-------------')