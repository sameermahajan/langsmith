#!/usr/bin/env python
# coding: utf-8

# # RAG Application

# ![Simple RAG](../../images/simple_rag.png)
# 
# In this notebook, we're going to set up a simple RAG application that we'll be using as we learn more about LangSmith.
# 
# RAG (Retrieval Augmented Generation) is a popular technique for providing LLMs with relevant documents that will enable them to better answer questions from users. 
# 
# In our case, we are going to index some LangSmith documentation!
# 
# LangSmith makes it easy to trace any LLM application, no LangChain required!

# ### Setup

# Make sure you set your environment variables, including your OpenAI API key.

# In[ ]:

# Or you can use a .env file
from dotenv import load_dotenv
load_dotenv(dotenv_path="./.env", override=True)


# ### Simple RAG application

# In[12]:


from langsmith import traceable
from openai import OpenAI
from typing import List
import nest_asyncio
from utils import get_vector_db_retriever

MODEL_PROVIDER = "ollama"
MODEL_NAME = "llama3.1"
APP_VERSION = 1.0
RAG_SYSTEM_PROMPT = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the latest question in the conversation. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
"""

openai_client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
nest_asyncio.apply()
retriever = get_vector_db_retriever()

"""
retrieve_documents
- Returns documents fetched from a vectorstore based on the user's question
"""
@traceable(run_type="chain")
def retrieve_documents(question: str):
    return retriever.invoke(question)

"""
generate_response
- Calls `call_openai` to generate a model response after formatting inputs
"""
@traceable(run_type="chain")
def generate_response(question: str, documents):
    formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    messages = [
        {
            "role": "system",
            "content": RAG_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Context: {formatted_docs} \n\n Question: {question}"
        }
    ]
    return call_openai(messages)

"""
call_openai
- Returns the chat completion output from OpenAI
"""
@traceable(run_type="llm")
def call_openai(
    messages: List[dict], model: str = MODEL_NAME, temperature: float = 0.0
) -> str:
    return openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

"""
langsmith_rag
- Calls `retrieve_documents` to fetch documents
- Calls `generate_response` to generate a response based on the fetched documents
- Returns the model response
"""
@traceable(run_type="chain")
def langsmith_rag(question: str):
    documents = retrieve_documents(question)
    response = generate_response(question, documents)
    return response.choices[0].message.content


# This should take a little less than a minute. We are indexing and storing LangSmith documentation in a SKLearn vector database.

# In[ ]:


question = "What is LangSmith used for?"
ai_answer = langsmith_rag(question, langsmith_extra={"metadata": {"website": "www.google.com"}})
print(ai_answer)


# ### Let's take a look in LangSmith!
