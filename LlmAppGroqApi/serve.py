from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
import os
from langserve import add_routes
from dotenv import load_dotenv
import traceback

# Load environment
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# Define the model
llm = ChatGroq(model_name="mixtral-8x7b-32768", groq_api_key=groq_api_key)

# Define prompt and output
prompts = ChatPromptTemplate.from_messages([
    ("system", "Translate the following english text to {language}."),
    ("human", "{text}")
])
output_parser = StrOutputParser()

# Build chain
chain = prompts | llm | output_parser


# Create FastAPI app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain Runnable Interfaces"
)

# Add routes
add_routes(app, chain, path="/chain")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8007)

