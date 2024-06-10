# Standard library import
import logging
from decouple import config
import os

# Third-party imports
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from fastapi.responses import PlainTextResponse

from langchain import hub
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ChatMessageHistory

# Local imports
from ensemble import ensemble_retriever_from_docs
from rag_chain import make_rag_chain, get_question
from local_loader import load_pdf_files
from basic_chain import basic_chain, get_model
from splitter import split_documents
from vector_store import create_vector_db
from memory import create_memory_chain


# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid = config("TWILIO_ACCOUNT_SID")
auth_token = config("TWILIO_AUTH_TOKEN")
openai_api_key = config("OPENAI_API_KEY")
twilio_number = config('TWILIO_NUMBER')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_retriever(openai_api_key=None):
    docs = load_pdf_files()
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
    return ensemble_retriever_from_docs(docs, embeddings=embeddings)


def get_chain(openai_api_key=None, huggingfacehub_api_token=None):
    model = get_model("ChatGPT")
    chat_memory = ChatMessageHistory()
    ensemble_retriever = get_retriever(openai_api_key=openai_api_key)
    output_parser = StrOutputParser()
    rag_chain = make_rag_chain(model, ensemble_retriever) 
    chain = create_memory_chain(model, rag_chain, chat_memory) | output_parser
    
    return chain

def run_rag_query(query):
    """Helper function to run RAG Query

    """
    memory_chain = get_chain(openai_api_key=openai_api_key)
    response = memory_chain.invoke(
            {"question": query},
            config={"configurable": {"session_id": "foo"}}
        )
    return response

def search_wikipedia(query):
    """Search Wikipedia through the LangChain API
    and use the OpenAI LLM wrapper and retrieve
    the agent result based on the received query
    """
    prompt = hub.pull("hwchase17/react")
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    tools = load_tools(["wikipedia"], llm=llm)
    agent = create_react_agent(llm=llm, tools=tools,prompt=prompt, )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    output = agent_executor.invoke({"input": "{}".format(query)})

    return output['output']

def main():
    chain = get_chain(openai_api_key=openai_api_key)
    questions = [
        "Are there any disease outbreaks in Zambia?",
        "When did Anthranx start in Zambia?"]
    for q in questions:
        print("\n--- QUESTION: ", q)
        output = chain.invoke(q)
        print('OUTPUT TYPE==>', type(output))
        print("* RAG:\n", chain.invoke(q))

if __name__ == '__main__':
    # this is to quite parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()