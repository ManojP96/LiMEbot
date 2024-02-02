import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
# from langchain.llms import HuggingFaceHub
# from htmlTemplates import css, bot_template, user_template
from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message
# from PyPDF2 import PdfReader
# from utilities import set_header,initialize_data,load_local_css,load_authenticator
import pandas as pd
from PIL import Image
import numpy as np
# from PyPDF2 import PdfReader
# import io
# import pickle 
from langchain.llms.bedrock import Bedrock
# import boto3
# from AWS_credentials import *
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import Docx2txtLoader
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from chatbot_utils import *
import pickle
image = Image.open(r'lime_img.png')


def chatbot1(section,disable):
    if "boto3_bedrock" not in st.session_state:
        st.session_state['boto3_bedrock']=BedrockLLM.get_bedrock_runtime_client()

    st.session_state['boto3_bedrock']

    # - create the Anthropic Model
    if 'llm' not in st.session_state:
        st.session_state['llm']=Bedrock(model_id="meta.llama2-13b-chat-v1", client=st.session_state['boto3_bedrock'], model_kwargs={'max_gen_len':500,'temperature':0.2})
    
        st.session_state['bedrock_embeddings']=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=st.session_state['boto3_bedrock'])
        bedrock_embeddings =st.session_state['bedrock_embeddings']
    # path=r"C:\Users\SrishtiVerma\LimeLLM" 
    if 'vectorstore_faiss' not in st.session_state:

        try :
            st.session_state["vectorstore_faiss"] = FAISS.load_local("/FAISS_new_saved_embeddings", bedrock_embeddings)
        except :
            doc_path = "Technical Documentation (1).docx"
            loader = Docx2txtLoader(doc_path)
            document = loader.load()

            text_splitter = CharacterTextSplitter(separator='\n\n', chunk_size=500)
            docs = text_splitter.split_documents(document)
            st.session_state["vectorstore_faiss"] = FAISS.from_documents(
                docs,
                st.session_state['bedrock_embeddings'],
            )
            # save  the embeddings to local folder
            st.session_state["vectorstore_faiss"].save_local("./FAISS_new_saved_embeddings")

    #wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=st.session_state["vectorstore_faiss"])


    if 'memory' not in st.session_state:
        
        st.session_state['memory'] = ConversationBufferMemory(k=3, memory_key='history', return_messages=True)
    
    prompt_template = """

    Human: You're a chatbot talking to a human. 
    Use the following pieces of context to provide a concise answer to the question at the end. Start the answer without any greetings.
    If needed, provide a step-by-step process only if you can, otherwise provide a short and concise answer.
    If you don't know the correct answer, just say that you don't know, don't try to make up an answer.
    Don't ask any more questions.

    <context>
    {context}
    </context

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=st.session_state['llm'],
        chain_type="stuff",
        retriever=st.session_state["vectorstore_faiss"].as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        memory=st.session_state['memory'],
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )


    def handle_userinput(query):
            result = qa({"query":query}) # new
            st.session_state['history'].append((query, result["result"]))
            return result["result"]

            # st.session_state.conversation = initialize_conversation_chain(pdf_docs)
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hey!👋, Ask me anything about LiME or select a question from the dropdown below:"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey!👋"]
        
    container_head = st.container()
    
    col1, col2 = st.columns([1,4])
    image = Image.open(r'lime_img.png')
    with container_head:
        with col1:
            st.image(image)
        with col2:
            st.markdown("<h1 style='text-align: center; color:Black; font-size: 50px;'>LiBo</h1>", unsafe_allow_html=True)
    response_container = st.container(height=600, border= True)
    #container for the user's text input
    container = st.container(border = True)

    
    
    def clear_textbox_input():
        st.session_state['radio2'] = st.session_state.radio_selection
        st.session_state.input=''

    questions=pd.read_excel('questions.xlsx')
    questions=questions[questions['Section']==section]

    with container:
        question = st.selectbox('Select a Query', questions['Question'],index=0)
        user_text = st.text_area(label='',placeholder="Type your query or select a question from the drop-down above.", key='input')
        # st.session_state.user_text.append(user_text)
        
        

        
        if disable:
            cols1,cols2=st.columns([3,2])
            with cols1:
                st.radio('Interact with: ', ['Document','Input Data'], key='radio_selection',on_change=clear_textbox_input,horizontal=True) 
            with cols2:
                st.radio('', ['Model Output'], key='radio_selection1',horizontal=True,disabled=disable,help="Please create a model to unlock access to this chatbot") 
        else:
            st.radio('Interact with:', ['Document','Input Data','Model Output'], key='radio_selection',on_change=clear_textbox_input,horizontal=True) 

        columns=st.columns(2)

        with columns[0]:
            if st.button(label='Send'):
                if len(user_text)>0:
                    output = handle_userinput(user_text)
                    st.session_state['past'].append(user_text)
                    st.session_state['generated'].append(output)
                
                else:
                    output = handle_userinput(question)
                    st.session_state['past'].append(question)
                    st.session_state['generated'].append(output)
                
        with columns[1]:
            clear_chat=st.button('Clear Chat')


        if clear_chat:
            st.session_state['memory'].clear() 
            st.session_state['past']=["Hey!👋"]
            st.session_state['generated']=["Hey!👋, Ask me anything about LiME or select a question from the dropdown below:"]

        st.session_state['generated']=[item.replace('\n', ' ').strip()  for item in st.session_state['generated']]
        image = Image.open(r'lime_img.png')

        if st.session_state['generated']: 
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    st.chat_message("user").write(st.session_state["past"][i])
                    st.chat_message("assistant",avatar=np.array(image)).write(st.session_state["generated"][i])
                
               # st.write(st.session_state['generated'][i])
