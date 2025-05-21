# import all the neccessary modules
import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
# enviroment variable, os, groq, chatprompt template, create_stuff document chain, create retrieval chain,

# load enviroment variables, 
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
groq_api_key=os.getenv('GROQ_API_KEY')

# create groq model
llm = ChatGroq(model="Llama3-8b-8192")

# create chatprompt template

prompt = ChatPromptTemplate.from_template(
"""
You are a helpful assistance and use the following context to answer the questions.
please prorovide the answer strickly based on the context provided.
<context>
{context}
<context>
User question: {input}
"""
)
# vector embedding function 
# st.seesion_state.embedding, loader, docs, text_splitter, final_document(split_documents()), vectors=FAISS(final_docuemnts, embeddings)
# No need to return the final vectors since it is saved in the session.

def create_vector_embedding():
    if "vector" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings()
        st.session_state.loader = PyMuPDFLoader('pdf_repo')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_document=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS(st.session_state.final_documents, st.session_state.embeddings)


# display text input requesting for query
user_prompt=st.text_input("Enter your query from the research papaer")
#add button to create/ add embedding to the session.
if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector embedding completed!")
# After embedding completes and user provides input

if user_prompt:
    document_chain=create_stuff_documents_chain(llm, prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever, document_chain)

    start=time.process_time()
    response= retrieval_chain.invoke(user_prompt)
    print(f"Process completed Response time: {start-time.process_time()}")

    st.write(response['answer'])