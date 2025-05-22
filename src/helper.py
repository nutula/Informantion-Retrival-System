import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import google_palm
from langchain.vectorstores import FAISS
from langchain.chains import conversational_retrieval
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = page.extract_text()
    return

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    llm=google_palm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = conversational_retrieval.from_llm(llm=llm, retriver=vector_store.as_retriver(),memory=memory)
    return conversation_chain