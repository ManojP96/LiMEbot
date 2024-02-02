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

# st.set_page_config(
#   page_title="Data Validation",
#   page_icon=":shark:",
#   layout="wide",
#   initial_sidebar_state='collapsed'
# )
# # load_local_css('styles.css')
# # set_header()


def chatbot_model_output(disable):
    if "boto3_bedrock_3" not in st.session_state:
        st.session_state['boto3_bedrock_3'] = BedrockLLM.get_bedrock_runtime_client()

  
    if 'output_cluade_llm' not in st.session_state:
        st.session_state['output_cluade_llm'] = Bedrock(model_id="anthropic.claude-v2:1", client=st.session_state['boto3_bedrock_3'], model_kwargs={'temperature':0.2})
        bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=st.session_state['boto3_bedrock_3'])

        st.session_state["output_llm_memory"] = ConversationBufferMemory(k=3, memory_key='output_llm_history', return_messages=True)

    df = pd.read_csv("model_result_data.csv")
    output_llm_sel = df.columns.tolist()
    output_llm_str_sel=', '.join(output_llm_sel)

    output_llm_prompt_template = """

    Human: You are provided with a dataframe with the following columns - """ + output_llm_str_sel + """
    You are talking with a human who wants to ask queries on the data. Create pandas code to solve the queries.
    If you do not have an answer reply with 'I am sorry, I don't have this information.'.
    Only output the pandas code.


    Question: {question}

    Assistant:"""

    output_llm_PROMPT = PromptTemplate(template=output_llm_prompt_template, input_variables=["question"])
    output_llm_qa = LLMChain(llm=st.session_state['output_cluade_llm'], prompt=output_llm_PROMPT, memory=st.session_state["output_llm_memory"])


    # Function to handle user input and interact with the conversation chain
    def handle_userinput(query,df):
        ans = output_llm_qa({"question":query})
        pattern = r"```python\n(.*?)\n```"
        match = re.search(pattern, ans['text'], re.DOTALL)
        if match:
            extracted_code = match.group(1).strip()
        try:
            print(extracted_code)
            result_series = eval(extracted_code)
            if isinstance(result_series, pd.DataFrame):
                return result_series
            else:
                result_df = pd.DataFrame(result_series.tolist(), columns=['Feature_set'])
                return result_df
        except Exception as e:
            result_series = eval(extracted_code)
            if result_series:
                return result_series
            else:
                return 'Cannot be excuted'

            #st.session_state.conversation = initialize_conversation_chain(pdf_docs)
    if 'history_output' not in st.session_state:
        st.session_state['history_output'] = []

    if 'generated_output' not in st.session_state:
        st.session_state['generated_output'] = ["Hey!👋, Ask me anything about Data "]

    if 'past_output' not in st.session_state:
        st.session_state['past_output'] = ["Hey!👋"]
        
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

    if 'user_text_output' not in st.session_state:
        st.session_state.user_text_output=['']
    with container:
        user_text = st.text_area(label='',placeholder="Query: Type your query or select any option from the dropdown below", key='input_model')
        st.session_state.user_text_output.append(user_text)



        def clear_textbox_input():
            st.session_state['radio2'] = st.session_state.radio_selection
            st.session_state.input=''
            # st.session_state["output_llm_memory"].clear()
            # st.rerun()
        def reset():
            st.session_state.input_model=''

        if disable:
            cols1,cols2=st.columns(2)
            with cols1:
                st.radio('Interact with: ', ['Document','Input Data'], key='radio_selection',on_change=clear_textbox_input,horizontal=True) 
            with cols2:
                st.radio('', ['Model Output'], key='radio_selection1',horizontal=True,disabled=disable,help="Please create a model to unlock access to this chatbot") 
        else:
            st.radio('Interact with:', ['Document','Input Data','Model Output'], key='radio_selection',on_change=clear_textbox_input,horizontal=True) 
    
        #st.write('<style>div.row-widget.stRadio > div{tooltip:attr(data-help);}</style>', unsafe_allow_html=True)
        columns=st.columns(2)

        with columns[0]:
            if st.button(label='Send',on_click=reset):
                user_input = st.session_state.user_text_output[-2]
                output = handle_userinput(user_input,df)
                st.session_state['past_output'].append(user_input)
                st.session_state['generated_output'].append(output)

        with columns[1]:
            clear_chat=st.button('Clear Chat',on_click=reset)

        if clear_chat:
            st.session_state["output_llm_memory"].clear() # new
            st.session_state['past_output']=["Hey!👋"]
            st.session_state['generated_output']=["Hey!👋, Ask me anything about LiME "]

        # if 'Input Data' not in st.session_state:
        #     st.session_state['Input Data']=False
    
        # if st.session_state['Input Data']:
        #     st.markdown('You are interacting with **Input Data**')
        # if st.session_state['Output Data']:
        #     st.markdown('You are intearacting with **Output Data**')
        image = Image.open(r'lime_img.png')

        if st.session_state['generated_output']:
            with response_container:
                for i in range(len(st.session_state['generated_output'])):
                    st.chat_message("user").write(st.session_state["past_output"][i])
                    if isinstance(st.session_state["generated_output"][i], pd.DataFrame):
                        st.chat_message("user").dataframe(st.session_state["generated_output"][i])
                    else:
                        st.chat_message("assistant",avatar=np.array(image)).write(st.session_state["generated_output"][i])

                #   message(st.session_state["generated"][i], key=str(i),logo="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxAPDxEPEA4SEhAWEg0PDxUQDw8QEhARFREWFhUSExUYHSggGBolGxMVITEhJSkrLi4uFx8zODMtNygtLisBCgoKDg0OGhAQGyslICUtLjUtNTcvLS0tLS0tLS0tLS0tLS01LS0rLy0tLSstLS0tLS01LS0rLS0tLS0tLS0tLf/AABEIAOEA4QMBEQACEQEDEQH/xAAbAAEAAwEBAQEAAAAAAAAAAAAAAwQFAgEGB//EADkQAAIBAQQHBgUCBgMBAAAAAAABAgMEBREhEjFBUWFxkSIygaHB0SNCseHwE4IVUmKisvFDksIU/8QAGwEBAAMBAQEBAAAAAAAAAAAAAAMEBQYBAgf/xAA4EQEAAQMBBAYIBQQDAQAAAAAAAQIDBBEFEiExEyJBUWHRBjJxgZGhscEUI0Lh8DNScvEVQ2Ik/9oADAMBAAIRAxEAPwD9xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAILVaVTWLxezLeUM7aFrEt79es+x927c1zpCGxW/9RuLjg9azxxRS2XtqnNrm3NO7POOPOEl2xNuNdV03EAAAAAAAAAAAAAAAAAAAAAAAAAAAADmU0tZDcv0W41mXsRqr/8A2QbwU11Mqdq2blW7TXHxS9FVEazBXp6cXHevPYR5NmL1qq3Pb9SirdqiWJQqOE1Lc8/o0cXh36sXJpuf2zx+ktGumK6Jh9ImfqUTExrDIenoAAAAAAAAAAAAAAAAAAAAAAAAHMppENy/RbjWZexGqhabziso9p+XU5zN9IaKera4z8vj5LNvGmefBmV7RKfeeW5ZLocrk517Inrzw7uxdotU0ckRVfbXu2q5QwbzTw8Nh1+x8iq7Y0qnjTwUMijdq4KF4Q0aj44S6/fEwtrWujyavHj8VmxVrRDYu6ppUo8Fo9MjuNjX+lw7c90afDgoX6dK5WTTRAAAAAAAAAAAAAAAAAAAAAAHMppEVy9RRHGXsRqoWm84rKPafDV1OczfSC3R1bfGfl8fJZt40zzZle0Sn3nluWSOVyc69kT154d3YuUWqaOSIppAABeumXbkt6x6P7m7sG5pdqo74+n+1bKjqxLq945xfBr86ku3qOtRX7YfOLPOFi5ZdiS3Sx6r7Gv6MXNceunuq+sIsuOtEtE6VVAAAAAAAAAAAAAAAAAAAA8csCKu7TRze6Kdttn6aWWLeOH3MXaW1px6YmI58v8Aaa1Z35ZFe0ynreW5ZI43Jzr2R688O7sX6LVNHJEUkgAAAALV2v4i5S+hrbGnTKj2SgyPUWr37sefoae3o/Konx+yHF9aS433/wBnqTeitX9WP8fuZnY1Tr1IAAAAAAAAAAAAAAAAAOZSSIrl6iiOMvYjVEqyl3WnyeJQ/GxeiejmNPCdX3uTHN6RipWtFJvQnhlvTwT5mZkZeFXVNm7McO+OGvtTUUXIjepRTu+Es4S/9IqXNjWLsb1mrT5w+4yKqeFUKtWxVI7MV/Tn5GVf2Tk2uOmseHlzT036KlZmdNM0zpKbUPkAAFq7V8RcpfQ1tjRrlR7JQZHqLN7vsx5v6Gnt6fyqI8fshxfWl7ca7/7PUm9Faf6s/wCP3MyeTVOvUgAAAAAAAAAAAAAADmUkiK5dpo5y9iNVG03lGOUe0+Grqc9m7ft2+rb4z4efksW8aqrmzK9pnPW8tyyRyuVtC/kT154d3Z+67Rapo5Lt1xwjKb1ei/2bOxLe5Zru1dv2VsmdaopcWa3Sc8JPsvJZLLcQ4e1rleRpcnqzy8O59XLERRrHNFeVHRnpLVLPx2lfbOL0d7pI5VfXtfePXrTp3KsJtZptPg8DKt3a7c60TMJ5piea3SvGa14S55PqjVsbbv0cK9KvlKCrGpnlwWFa6VTKccOax80aMbRwsmNLtOnt80M2blHqvJXfCSxhLDx0kfNzY2Pdjes1afOHsZFVPrQq1bDUjsxX9OfkZd7ZGTb4xGseHknpv0SrPLWZtVM0zpMaJonVpXXQabm1hlguPE6TYuJXRM3q404aQp5FyJ6sOL3lnFcG+v8Aoi29X1qKPbL6xY4TKzcsexJ75fRGx6MW93Gqq76vpCHLnrxDROlVQAAAAAAAAAAAAOZSSI7l2miOMvYjVRtN5RjlHtPhq6nPZu37dvWm3xnw8/JYt41VXNmV7VOet5blkjlMraF/I9eeHdHJdotU08kJRSB7EazpA1LV8Oio7XhHrm/U6rNn8LgRbjnPD7yo2+vd1ZZysLzVj8ajh8y/yXv6nWU6bQwtP1R9Y81Gfyrngyjk5iYnSV4PAA9jJp4ptPg8CS3crtzrRMw8mmJ5rVK8JrXhJccn1NSztrIo4VaVfVBVj0zy4L1mtUamzCS2P0Zu4efZy50iNKo7J+ytctVULRoomJb6mlUlw7PT74nFbUvdJk1THZw+H7tGzTpRDZu+no0orhi/HM7vZFjocO3T4a/Hiz71W9XMrBpIgAAAAAAAAAA8csCOu7TRze6Kdstqp7MW9S9WYe0dsRjxpEcZ7PNNaszWya9qnPW8tyyX3OPyto38j1p4d0cv3X6LVNPJCUUgAAnsNPSqR3LtPwNDZdjpcmmOyOPwRX6t2iU161MZqO5eb/EXNuX969FuP0x85R41OlOvepGGsrV31tGeD1PJ89jNbZGV0N7dnlVw9/Ygv0b1Ovc6vKjoy0lql9dpJtnF6O70kcqvq8x69adO5TMVYAAACexY/qRw3+WGZf2ZvfiqN3v+Xaivabkte0VdCLluWXPYdflX4sWqrk9n17FCineqiGNZqWnOMd7z5bTjsHHnKyabffPH2c5aFyrcomX0aP1CI0jSGS9PQAAAAAAAAAZN4W6am4RySwz2vLHocdtnbORavzZtdWI049s/su2LFM070rdnqqcVJePB7UaWNkRkWouR28/agro3atGXeSf6jx3Rw5Ye+Jyu2IqjKmZ5aRp7NPPVdx9NxVMpOAAAGldUMIym+Xgs3+cDpth2oot13p/kRzU8mrWYpZ9WelJy3ts5+/dm7dqrntlaop3aYhyQvoPRqw+NSw+Zf5LUzrKJjaGFpPrR9Y81GfyrjKOUmJidJXonUPkAAGpdlnwWm9b1cEdVsbC6Ojpquc8vZ+6lkXNZ3YQ3pXxegtS18yntrL364s08o5+39kmNRpG9Kxc1DBOo9uUeW1/m41vRrC3aJyKo58I9nb/PBFlXNZ3YaZ1SmAAAAAAAAAAGde1l0lppZrXxX2Oa9IdndNb6eiOtTz8Y/Zaxru7O7PaoWG06EsH3Xr4cTm9l534e5u1erPPw8fNavW9+NY5tK12dVI8dcX+bDos7CpyrenbHKf52Kdq5NEsWcWm01g1rOLuW6rdU01RpMNGJiY1h4fD0AHsRrwGpafh0FHa0o+LzfqdVl/8Ay4EW45zGnx5qNvr3dWWcqvB4AFq762hPB6nk+exmrsnK6G/uzyq4eSC/RvU69zq86OjLSWqX12ku2cXo7vSRyq+rzHr1p07lMxVgPRbsNk03pSXZX9z3cjY2Xs7pqukuR1Y+f7K967uxpHNfttp/Tjl3nq9zc2jmxjW+HrTy81a1b358GXZqLqTUfGT3LazmcDDrzb8Ud/GZ7o7V25XFunV9DCCiklqWSP0y3bpt0RRTGkRyZUzrOsuj7eAAAAAAAAAAAEjDvKx6D0orsP8Ate7kcDtvZU41fS246k/KfLuaOPe343Z5vbBbNHsSeWx7uD4Hmy9p9Hpauzw7J7vCfB5es69alctdlVRbpbH6M187Z9GVTryq7J80Fu7NE+DIq0nB4SWD+vI5C/j3LFW7cjT7r9NcVRrDggfSexU9KpFbO8/D8RobMsdLk0x2Rxn3Ir1W7RKa9amMlHcvN/iLu3L+9di3HZHzlHjU6UzKkYSyAAB6Nan8alg+9q/ctp1trTaGFuzz+8KFX5VzgzJ0pReDi8eRzN3FvW6t2qmV2mumY1iVqyWFyzmsI7tTfsamBsiq5O/ejSO7tnyQXb8RwpX7RXjSj5RSN7KyrWJb4+6FaiibkseUpVJb5PJeyOPqqu5d7vqnkvxFNunwblhsqpxw+Z5yfpyP0HZezqcKzu/qnnP29kM27dm5VqsmmiAAAAAAAAAAAAA5nFNYNYp5M+Llum5TNFUaxL2JmJ1hiW6wunms4eceDOB2tsavEmblvjR849vm0bN+K+E83lktrhk84+a5EWBtWqx1LnGn5wXbEVcY5tLCFWOyS+nszpJixl2+yqJ/nuVOtRPcpzuzPKeXFGPc2DE1dSvh4wsRlcOMLVlsqprLNvWzUwsCjFp4cZnnKC5dmueLLt0Wqksd+K5HL7Uoqpyq9e36LtiYmiNEBnpQAAPRsXdRcYZ628cN2R2WycaqzY63OZ1Z9+uKquC0aaFUtVujDJZy8lzMrN2rbsdWjjV8o9vkmt2Jq4zyZnaqS2yk/wA6HM/n5l7tqqn+e6F3q26fBs2CxKmsXnN63u4I7rZOyKMOneq41zznu8I/nFn3r03J8Fw2UAAAAAAAAAAAAAAAB40eTETGkjMtl2fNT/6+xym0vR2KtbmNw/8APl5LlrK04Vs6MpU5bYvb90cvTVfxLmnGmpbmKa4716jeWya8V7G3jbcjlej3x5K1eNP6V2nXjLuyT+vQ2rOVZvRrRVEq9VFVPOHNezxn3lyepo+cnDtZEaXI83tFyqjkrO7I7JS8mZk7Bs9lU/JN+Jq7nn8Lj/O+iPn/AIG3/fPyPxVXc7jdsFrcnza9CajYmPTz1n3+T5nJrlPChCGailxfuy9bxcexxppiPFHNdVXOUda3wjqek+HuVr+1se1wid6fDzfdNiupn17dOeXdW5erMHK2tev8I6seHmtUWKaefF5ZbFOpqWEd71eG8YGyMjLnWI0p75+3eXL9NHtbVlssaawSz2t62d1g7Os4dOluOPbPbLPuXaq54py+jAAAAAAAAAAAAAAAAAABFWs8ZrCUU/quTKuThWMmndu0xP1+L7orqonWJZ1e6X8kseEvc5jK9GKo42KvdPmtUZf90KNWzTjrg+eGK6owb+zsqxPXon284+MLNN2irlLyFomtU31xI7ebkW/Vrn+e17NuiexKrfU/mx5pFmnbGVHbE+58fh6Hv8Qqb10Pv/msrvj4PPw9DiVtqP534JIgr2plVfr+j6ixRHYj7U380n4yK/59+f1VT75ffVp8FmjdtSWtaK46+hq43o/l3eNUbsePkhryaKeXFoWe7YRzfafHV0Okw9gY1jSqvrz48vh/tVrya6uXBcwNyI0V3p6AAAAAAAAAAAAAAAAAAAAAAACOdGMtcU+aTK9zEsXfXoifdD6iuqOUonYKT+ReGKKlWxsGrnbj5w+4v3I7Xn8Ppfyecvc+I2HgR/1/OfN7+Iud7uNjpr/jj0xLFGzMSj1bdPwfM3a57UyilqWHIuU0U08KY0R66vT6AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/2Q==")
                    #st.chat_message("assistant",avatar=np.array(image)).dataframe(st.session_state["generated_output"][i])
                    #with st.chat_message("assistant",avatar=np.array(image)):
                    #    st.dataframe(st.session_state["generated_output"][i])
                    #    if isinstance(st.session_state["generated_output"][i], str):
                    #        st.write(st.session_state["generated_output"][i])
                    #    else:
                    #        st.dataframe(st.session_state["generated_output"][i])
