#!/usr/bin/env python
import ollama
import bs4
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

# Much of this code is generated with help of AI bots

# Load URL
kb_loader = WebBaseLoader("https://mnjagadeesh.net")
docs = kb_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                               chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# use mistral model with vector store
embeddings = OllamaEmbeddings(model="mistral")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# retriever handle
retriever = vectorstore.as_retriever()

# put it all togather
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the Ollama LLM function
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Define the RAG chain
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

# Send in question
print("Press 'q' to quit.")
while True:
    user_input = input("Ask Question: ")
    if user_input.lower() == 'q':
        print("Quitting...")
        break
    result = rag_chain(user_input)
    print(result)

