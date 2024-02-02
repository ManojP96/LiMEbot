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


def chatbot_copy(section,disable):
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
        

    response_container = st.container()
    #container for the user's text input
    container = st.container()

    
    def clear_chat2():
        st.session_state['memory'].clear() 
        st.session_state['past']=["Hey!👋"]
        st.session_state['generated']=["Hey!👋, Ask me anything about LiME or select a question from the dropdown below:"]

    def clear_textbox_input():
        st.session_state['radio2'] = st.session_state.radio_selection
        st.session_state.input=''

    def clear_content():
        st.session_state.input=''



    questions=pd.read_excel('questions.xlsx')
    questions=questions[questions['Section']==section]

    if 'user_text' not in st.session_state:
        st.session_state.user_text=['']

    user_text = st.text_area(label='',placeholder="", key='input')
    
    st.session_state.user_text.append(user_text)

    question = st.selectbox('Select a Query', questions['Question'],index=0)
    
    
    if len(st.session_state.user_text[-2])>0:
        user_input=st.session_state.user_text[-2]
        output = handle_userinput(user_input)
    else:
        user_input=question
        output = handle_userinput(user_input)
    
    if disable:
        cols1,cols2=st.columns(2)
        with cols1:
            st.radio('Interact with: ', ['Document','Input Data'], key='radio_selection',on_change=clear_textbox_input,horizontal=True) 
        with cols2:
            st.radio('', ['Model Output'], key='radio_selection1',horizontal=True,disabled=disable,help="Please create a model to unlock access to this chatbot") 
    else:
        st.radio('Interact with:', ['Document','Input Data','Model Output'], key='radio_selection',on_change=clear_textbox_input,horizontal=True) 

    columns=st.columns(2)

    with columns[0]:

        if st.button(label='Send'):
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
    with columns[1]:
        clear_chat=st.button('Clear Chat')
 

    if clear_chat:
        st.session_state['memory'].clear() 
        st.session_state['past']=["Hey!👋"]
        st.session_state['generated']=["Hey!👋, Ask me anything about LiME or select a question from the dropdown below:"]

    st.session_state['generated']=[item.replace('\n', ' ').strip()  for item in st.session_state['generated']]

    if st.session_state['generated']: 
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user',logo='data:image/jpeg;base64,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')
                message(st.session_state["generated"][i], key=str(i),logo="data:image/jpeg;base64,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")
               
               # st.write(st.session_state['generated'][i])
