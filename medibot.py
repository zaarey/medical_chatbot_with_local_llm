import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

#step1: load the faiss db

db_faiss_path= 'vectorstore/db_faiss'
@st.cache_resource
def get_vectorstore():
    embedding_model= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db= FAISS.load_local(db_faiss_path, embedding_model, allow_dangerous_deserialization=True)
    return db

#step2: load the prompt function 

def get_custom_prompt(custom_prompt):
    prompt= PromptTemplate(
        template= custom_prompt, 
        input_variables= ['context', 'question']
    ) 
    return prompt

#step3: load llms 

HF_token= os.getenv("HUGGINGFACEHUB_API_TOKEN")
repo_id_key= 'mistralai/Mistral-7B-Instruct-v0.3'

def load_llm(repo_id_key):
    llm= HuggingFaceEndpoint(
        temperature= 0.5,
        repo_id= repo_id_key,
        huggingfacehub_api_token= HF_token,
        max_new_tokens= 512,
        task="text-generation"
    )
    return llm 


def main():
    st.title("Ask Chatbot! ")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []  
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
        

    prompt= st.chat_input("Write your prompt here...")    

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        custom_prompt= """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """

        #step4: load the qa_chain

        try: 
            vectorstore= get_vectorstore()
            if vectorstore is None:
                st.error('cannot load vector store')

            qa_chain= RetrievalQA.from_chain_type(
                llm= load_llm(repo_id_key),
                retriever= vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type= 'stuff',
                chain_type_kwargs={'prompt': get_custom_prompt(custom_prompt)})

            #step5: inovoke the chain to get the response

            response= qa_chain.invoke({'query': prompt})
            result= response['result']
            data_source= response['source_documents']
            result_to_show= result + " Source Docs: "+ str(data_source)
    

            #response="Hi, I am Medibot."
            st.chat_message('assistant').markdown(result_to_show, unsafe_allow_html=True)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e: 
            st.error(f'Error: , {str(e)}')

if __name__ == "__main__":
    main()
