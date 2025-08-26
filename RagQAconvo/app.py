import streamlit as st
import os
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
from dotenv import load_dotenv
load_dotenv()

# load environ variables
os.environ['OLLAMA_API_KEY'] = os.getenv('OLLAMA_API_KEY')
os.environ["USER_AGENT"] = "GenAI-with-langchain/1.0"
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

# set up streamlit

st.title("Conversational RAG with pdf uploads and chat history")
st.write("Upload Pdf's and chat with their content")

api_key = st.text_input("Enter your groq api key:",type = "password")

# check if groq api key is provided

if api_key:
    llm = ChatGroq(groq_api_key = api_key,model_name = "Gemma2-9b-It")

    # chat interface
    session_id = st.text_input("Session ID", value="default_session")

    # statefully manage the chat history

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose A Pdf File", type = "pdf", accept_multiple_files= True)

    # process uploaded pdf's

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf,"wb") as file:
                # write the content of upload file in temppdf
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            # load the file
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        # split the documents and convert text into vectors
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 50)
        split_docs = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(split_docs,
        embedding=embeddings,
        persist_directory="chroma_store")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history,"
            "formulate a standalone question which can be understood"
            "without the chat history. Do not answer the question,"
            "just reformulate it if needed and otherwise return as is it."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        # history aware retreiver
        history_aware_retreiver = create_history_aware_retriever(llm,retriever,prompt)
        # Answer question prompt
        system_prompt = (
            "You are a helpful assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer the question."
            "If you don't know the answer, just say that you don't know."
            "Use three sentences or less to answer the question."
            "\n\n"
            "{context}\nQuestion: {input}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(question_answer_chain,history_aware_retreiver)

        def get_session_history(session : str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        coneversational_rag_chain = RunnableWithMessageHistory(rag_chain,get_session_history,input_messages_key="input",history_messages_key="chat_history",output_messages_key="answer")

        user_input = st.text_input("Your Question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = coneversational_rag_chain.invoke(
                {"context": docs,
                "input": user_input},
                config = {
                    "configurable" : {"session_id" : session_id}
                }
            )

            st.write(st.session_state.store)
            st.success("Assistant:", response['answer'])
            st.write("chat_history:",session_history.messages)

else:
    st.warning("Please enter your groq api key: ")