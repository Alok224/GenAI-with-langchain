🧠 Gen AI Chatbot with Search Engine Agents
📌 Overview

This is a Generative AI Chatbot built using LangChain and Streamlit.
The chatbot is capable of answering user queries by leveraging multiple knowledge sources and intelligently combining them with a custom RAG (Retrieval-Augmented Generation) pipeline.

The system does more than just generate responses – it interacts with external tools (search agents) to fetch accurate and up-to-date information before generating answers.

🚀 Features

🔍 Tool-Enhanced Search

-> Arxiv Tool → Retrieves research papers and scientific content.

-> Wikipedia Tool → Fetches encyclopedic knowledge.

-> Custom RAG Tool → My own tool to retrieve domain-specific knowledge.

🤖 Intelligent Conversational Agent

-> Combines LLM power with search tools to give context-aware answers.

-> Reformulates questions when needed (history-aware retriever).

💾 Session State Management

-> Built with Streamlit → maintains conversation history using st.session_state.

-> Ensures smooth multi-turn conversation without losing context.

📚 Context-Aware Responses

-> Integrates retrieved documents into the response.

-> If the answer is not found → responds gracefully with “I don’t know.”

🏗️ Tech Stack

-> Python

-> LangChain (Agents, Prompts, RAG pipeline)

-> Streamlit (Frontend + Session state)

-> Arxiv API, Wikipedia API (External tools)

-> Custom RAG retriever

⚙️ How It Works

-> User enters a query in the Streamlit UI.

-> Chatbot checks conversation history (st.session_state).

-> Query is passed through history-aware retriever.

-> Depending on the query, the agent decides whether to:

-> Use Arxiv for research queries.

-> Use Wikipedia for general knowledge.

-> Use Custom RAG for domain-specific knowledge.

-> Retrieved context is passed to the LLM for final response.

-> Response is displayed in chat format.