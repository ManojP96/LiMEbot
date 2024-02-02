import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
import pandas as pd
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from chatbot_utils import * 
import re
import sys
from io import StringIO
from PIL import Image
import numpy as np
image = Image.open(r'lime_img.png')
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display

# st.set_page_config(
#   page_title="Data Validation",
#   page_icon=":shark:",
#   layout="wide",
#   initial_sidebar_state='collapsed'
# ) 
# # load_local_css('styles.css')
# # set_header()

def chatbot_input_data(disable):

    if 'boto3_bedrock_2' not in st.session_state:
        st.session_state['boto3_bedrock_2']=BedrockLLM.get_bedrock_runtime_client()

    if 'llm_2' not in st.session_state:
        st.session_state['llm_2'] = Bedrock(model_id="anthropic.claude-v2:1", client=st.session_state['boto3_bedrock_2'], model_kwargs={'temperature':0.2})
        st.session_state['llm_2']
        #bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=st.session_state['boto3_bedrock_2'])

    memory_input = ConversationBufferMemory(k=3, memory_key='history_input', return_messages=True)

    df = pd.read_excel("data.xlsx")
    sel = df.columns.tolist()
    str_sel=', '.join(sel)

    prompt_template_1 = """

    Human: You are provided with a dataframe with the following columns - """ + str_sel + """
    You are talking with a human who wants to ask queries on the data. Create pandas code to solve the queries.
    If you do not have an answer reply with 'I am sorry, I don't have this information.'.
    Only output the pandas code. The output pandas code should always have a display statement to show the results.


    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(template=prompt_template_1, input_variables=[ "question"])
    qa = LLMChain(llm=st.session_state['llm_2'], prompt=PROMPT, memory = memory_input)


    # Function to handle user input and interact with the conversation chain
    def handle_userinput(query,df):
        ans = qa({"question":query})
        pattern = r"```python\n(.*?)\n```"
        match = re.search(pattern, ans['text'], re.DOTALL)
        if match:
            extracted_code = match.group(1).strip()
            try:
                # st.write("Extracted Code:")
                # st.code(extracted_code, language="python")

                # Capture standard output
                original_stdout = sys.stdout
                sys.stdout = StringIO()

                # Execute the code and capture the result
                result = None
                try:
                    exec(extracted_code)
                    result = sys.stdout.getvalue()

                finally:
                    # Restore the original standard output
                    sys.stdout = original_stdout

                # Display the result and print output
                # st.write("Result:")
                # st.text(result)
                return result
            except Exception as e:
                # st.write(e)
                return 'Cannot be executed'

            # st.session_state.conversation = initialize_conversation_chain(pdf_docs)
    if 'history_input' not in st.session_state:
        st.session_state['history_input'] = []

    if 'generated_input' not in st.session_state:
        st.session_state['generated_input'] = ["Hey!👋, Ask me anything aboutInput Data "]

    if 'past_input' not in st.session_state:
        st.session_state['past_input'] = ["Hey!👋"]
    container_head = st.container()
    
    col1, col2 = st.columns([1,4])
    image = Image.open(r'lime_img.png')
    with container_head:
        with col1:
            st.image(image)
        with col2:
            st.markdown("<h1 style='text-align: center; color:Black; font-size: 50px;'>LiBo</h1>", unsafe_allow_html=True)
    response_container = st.container(height=600, border= True)
    container = st.container(border = True)

    
    with container:
        user_text = st.text_area(label='',placeholder="Query: Type your query", key='input')
            
        def clear_textbox_input():
            st.session_state['radio2'] = st.session_state.radio_selection
            st.session_state.input=''
            memory_input.clear()
        #tooltip_text = ['Ask any questions about functionality in LiME', 'Interact with uploaded data', 'Interact with the Model Results']
        if disable:
            cols1,cols2=st.columns(2)
            with cols1:
                st.radio('Interact with: ', ['Document','Input Data'], key='radio_selection',on_change=clear_textbox_input,horizontal=True) 
            with cols2:
                st.radio('', ['Model Output'], key='radio_selection1',horizontal=True,disabled=disable,help="Please create a model to unlock access to this chatbot") 
        else:
            st.radio('Interact with:', ['Document','Input Data','Model Output'], key='radio_selection',on_change=clear_textbox_input,horizontal=True) 
        
        def clear_content():
            st.session_state.input=''
            
        def clear_text():
            st.session_state.test1 = st.session_state["input"] 
            st.session_state["input"] = ''

        columns=st.columns(2)
        
        
        with columns[0]:
            submit_button = st.button(label='Send',on_click=clear_text)

        with columns[1]:
            clear_chat=st.button('Clear Chat')
        
        if "clear_chat" not in st.session_state:
            st.session_state.clear_chat = False

        if clear_chat:
            memory_input.clear() 
            st.session_state['past_input']=["Hey!👋"]
            st.session_state['generated_input']=["Hey!👋, Ask me anything about Input Data "]

        if submit_button:
            user_text_ = st.session_state.test1
            output = handle_userinput(user_text_,df)
            st.session_state['past_input'].append(user_text_)
            st.session_state['generated_input'].append(str(output))
            print(user_text_)
            print(output)


        image = Image.open(r'lime_img.png')
        if st.session_state['generated_input']:
            with response_container:
                for i in range(len(st.session_state['generated_input'])):
                    st.chat_message("user").write(st.session_state["past_input"][i])
                    st.chat_message("assistant",avatar=np.array(image)).write(st.session_state["generated_input"][i])
