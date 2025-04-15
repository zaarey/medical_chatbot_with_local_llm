# import required libraries 

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 

# as i'm not using pipenv as the environment manager, we'll do this

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# step 1: load raw data 

data_path= "data/"
def load_pdf_files(data):
    loader= DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents= loader.load()
    return documents

documents= load_pdf_files(data=data_path)
#print("length of the document :", len(documents))

# step 2: create chunks 

def create_chunks(extracted_data):
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size= 500,
        chunk_overlap= 50)
    text_chunk= text_splitter.split_documents(extracted_data)
    return text_chunk

text_chunk= create_chunks(extracted_data=documents)
#print("length of chunks is: ", len(text_chunk))

# step 3: create vector embeddings 

def create_embedding_model():
    embedding_model= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embedding_model

embedding_model= create_embedding_model()

# step 4: store embeddings in FAISS 
db_faiss_path= 'vectorstore/db_faiss' 

db= FAISS.from_documents(text_chunk, embedding_model)
db.save_local(db_faiss_path)