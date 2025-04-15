# import the dependencies 

import os 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS 
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

## as our virtual environment manager is not pipenv, do this 
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# step1: setup llm 

repo_id_key= 'mistralai/Mistral-7B-Instruct-v0.3'

def load_llm(repo_id_key):
    llm= HuggingFaceEndpoint(
        temperature= 0.5,
        repo_id= repo_id_key,
        huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        max_new_tokens= 512,
        task="text-generation"
    )
    return llm 


# step2: load FAISS

custom_prompt= """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def get_custom_prompt(custom_prompt):
    prompt= PromptTemplate(
        template= custom_prompt, 
        input_variables= ['context', 'question']
    ) 
    return prompt


db_faiss_path= 'vectorstore/db_faiss'
embedding_model= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db= FAISS.load_local(db_faiss_path, embedding_model, allow_dangerous_deserialization=True)

# step3: create chain 
from langchain.chains import RetrievalQA

qa_chain= RetrievalQA.from_chain_type(
    llm= load_llm(repo_id_key),
    retriever= db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type= 'stuff',
    chain_type_kwargs={'prompt': get_custom_prompt(custom_prompt)}
)

# step4: invoke chain
user_query= input("Write your query here:  ")
try:
    response= qa_chain.invoke({'query': user_query})
    print('response ', response['result'])
    print('data_source ', response['source_documents'])
except Exception as e:
    print("⚠️ An error occurred:", str(e))





