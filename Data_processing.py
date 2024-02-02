import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from Eda_functions import format_numbers,line_plot,summary
import numpy as np
from utilities import set_header,load_local_css,load_authenticator,initialize_data
import pickle
import time
from Data_prep_functions import *
from streamlit_float import *
from Chatbot1_input_data import *
#from Chatbot_document import *
from Chatbot_model_output import *
from chatbot_document_copy import * 
# from Data_prep_functions import column_name,Categorization


st.set_page_config(
    page_title="Data Preprocessing",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state='collapsed'
)
# Load CSS and set header (assuming these functions exist)
load_local_css('styles.css')
set_header()


for k, v in st.session_state.items():
    if k not in ['logout', 'login','config'] and not k.startswith('FormSubmitter'):
        st.session_state[k] = v

authenticator = st.session_state.get('authenticator')

if authenticator is None:
    authenticator = load_authenticator()
html_code = '''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                header {
                    background-color: #333;
                    color: white;
                    padding: 10px;
                    text-align: center;
                }

                .scrollable-chatbot {
                    max-height: 1000px;  /* Adjust the maximum height as needed */
                    overflow-y: auto;
                    padding: 10px;  /* Optional padding for better aesthetics */
                    scrollbar-width: thin;  /* For Firefox */
                    scrollbar-color: #888 #f0f0f0;  /* For Firefox */
                }

                .scrollable-chatbot::-webkit-scrollbar {
                    width: 8px;  /* For Chrome, Safari, and Opera */
                }

                .scrollable-chatbot::-webkit-scrollbar-thumb {
                    background-color: #888;  /* For Chrome, Safari, and Opera */
                }

                div:has( > .element-container div.floating) {
                    display: flex;
                    flex-direction: column;
                    position: fixed;
                }

                div.floating {
                    height: 0%;
                }
            </style>
        </head>
        <body>
            <header>
                <h1>Data Preprocessing Chatbot</h1>
            </header>

            <!-- Your other HTML content goes here -->

        </body>
        </html>
        '''   
name, authentication_status, username = authenticator.login('Login', 'main')
auth_status = st.session_state['authentication_status']

if auth_status:
    authenticator.logout('Logout', 'main')
    
    is_state_initiaized = st.session_state.get('initialized',False)

    # if not is_state_initiaized:
    #     initialize_data()
    # scenario = st.session_state['scenario']
    # raw_df = st.session_state['raw_df']

    if 'input_selected' not in st.session_state:
        st.session_state['input_selected']=False

    if "output_selected" not in st.session_state:
        st.session_state['output_selected']=False

    if 'radio2' not in st.session_state:
        st.session_state['radio2']=None

    if 'disabled' not in st.session_state:
        st.session_state['disabled']=True

    cols1,cols2=st.columns([9,2])        
    float_init()

    with cols2:
        container = st.container(border=True)

    with container:
        if st.session_state['radio2']=='Input Data':
            chatbot_input_data(st.session_state['disabled'])
        elif st.session_state['radio2']=="Model Output":
            chatbot_model_output(st.session_state['disabled'])
        else:
            chatbot1('Data Preprocessing',st.session_state['disabled']) 
        
        st.markdown('<div class="floating"></div>', unsafe_allow_html=True)

        #st.markdown(f'<script>{js}</script>', unsafe_allow_html=True) 
        #radios = st.radio('Select:', ['Input Data', 'Output Data'], key='radio_selection', on_change=radio)
        # st.write(st.session_state['radio2'])
        #container.float("bottom: 2rem;background-color: white;overflow-y: scroll;max-height: 1000px;border: 1px solid #add8e6;")

        
    with cols1:    
#@st.cache(allow_output_mutation=True)  # Cache data loading and preprocessing
        def load_and_process_data(uploaded_file):
            if uploaded_file is not None:
                file_extension = uploaded_file.name.split(".")[-1].lower()
                if file_extension == "csv":
                    data = pd.read_csv(uploaded_file)
                elif file_extension == "xlsx":
                    data = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file format. Please upload a CSV or Excel file.")
                    st.stop()

                prospects = pd.read_excel('data.xlsx', sheet_name='Prospects')
                data['Prospects'] = prospects['Prospects']
                data['Date'] = pd.to_datetime(data['Date'])
                
                
                #data.set_index('Date', inplace=True)
                return data
        # with col1:
        st.title('Data Preprocessing')


        if 'data_processed' not in st.session_state:
            st.session_state.data_processed = False

        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
        
        # with st.expander('Chat with Chatbot'):
        #     chatbot()
        
        if uploaded_file:
            with st.spinner('Categorizing Variables into Different Buckets'):
                time.sleep(0.5)  # Simulate processing time (remove this line in your actual code)
                st.success('Done!')   


            data = load_and_process_data(uploaded_file)


            
            variables = data.columns
            #st.text(variables)
            bucket_data=pd.read_excel('Variables.xlsx')

            if "old_categorized_data" not in st.session_state:
                st.session_state['old_categorized_data']=None




            bucket_data_dict = {
                row['Variable']: {
                    "VB": row["Variable Bucket"],
                    "MB": row["Master Bucket"],
                    "BB": row["Base Bucket"]
                }
                for index, row in bucket_data.iterrows()
            }  # storing the diferent buckts as dictionary

            Categorised_data= {}  # for the actual data

            for i in bucket_data_dict.keys():
                for j in data.columns:
                    if i.lower() in j.lower():
                        Categorised_data[j] = bucket_data_dict[i]
            # print('@@@@@@@@@')

            for i in data.columns:
                if i not in Categorised_data.keys():
                    Categorised_data[i]={'VB':'NC','MB':'NC','BB':'NC'}
            # print('@@@@@@@@@')
            # print(Categorised_data)

            bucket_data_VB={
                row['Variable Bucket']: {
                    "MB": row["Master Bucket"],
                    "BB": row["Base Bucket"]
                }
                for index, row in bucket_data.iterrows()
            }   #"Storing Variable_bucket as key value pair"

            renamed_variables = {}
            bucket_names =list(bucket_data_VB.keys())

            num_columns = 2
            num_rows = -(-len(variables) // num_columns)  # Ceiling division to calculate rows
            # variables = data.columns
            
            

            st.markdown('## Renaming and Classification')
            st.markdown('#### Instructions:')
            st.markdown('Use this section to rename variables if needed and reassign variable buckets. The bucket are initialised intuitively based on the variable name and might require changes. Click on the "Update Changes" button once complete and view the changes in the table generated after.')
            st.markdown('###### Rules for Variables Naming:')
            st.markdown('Please replace spaces with underscores and TBD')
            
            with open('old_Categorised_data.pkl', 'wb') as file:
                    pickle.dump(Categorised_data, file)
            #st.dataframe(data.head(2))
            with st.expander('Please check the checkbox to rename the variables and categorize the bucket if it has not been classified correctly',expanded=True):
            #with st.form("variable_renaming_form"):
                for row in range(num_rows):
                    cols = st.columns(num_columns)
                    for col_idx, col in enumerate(cols):
                        variable_idx = row * num_columns + col_idx
                        
                        if variable_idx < len(variables):
                            variable = variables[variable_idx]
                            # Checkbox for renaming variables
                            #col.markdown(f'**{variable}**', unsafe_allow_html=True)
                        rename_checkbox = col.checkbox(f'**{variable}**', key=f"{variable}_rename")
                        if rename_checkbox:
                                # Text input for new variable name
                            new_name = col.text_input(f"New Name for {variable}:", key=f"{variable}_new_name")
                            renamed_variables[variable] = new_name if new_name else variable  # Use new name if provided, else keep the original name
                            
                            if new_name:
                                    col.success('✅')
                        else:
                            renamed_variables[variable]=variable
                            # Selectbox for changing variable bucket

                        bucket_key = f"{variable}_bucket"
                        new_bucket_name = col.selectbox(f"Change Bucket Name for {variable}:", bucket_names, index=bucket_names.index(Categorised_data[variable]['VB']), key=bucket_key)
                        
                        Categorised_data[variable]['VB'] = new_bucket_name
                        Categorised_data[variable]['MB']=bucket_data_VB[new_bucket_name]['MB']
                        Categorised_data[variable]['BB']=bucket_data_VB[new_bucket_name]['BB']   
                        # Store the bucket name

                    #submitted_button = st.form_submit_button("Rename selected variables")  # Add the submit button here
            #if submitted_button:
            
                # st.write('renamed variables')  
                # st.write(renamed_variables)
                
            Categorised_data_copy = Categorised_data.copy()
            for old_key, value in Categorised_data_copy.items():
                if old_key in renamed_variables:
                    new_key = renamed_variables[old_key]
                    Categorised_data[new_key] = Categorised_data.pop(old_key)
            
                    # Iterate through the items in target_dict_copy
            def time_variable_selected():
                    st.success('Time variable selected')

            st.markdown('Select Time Variable')
            time_variables=[var for var in Categorised_data.keys() if Categorised_data[var]['VB']=="Date"]
            time_variable=st.selectbox('Variables',time_variables,on_change=time_variable_selected)
            
            st.markdown('### Base Price Selection')
            if st.checkbox('Select / Generate Base Price'):
                price_variables=[var for var in Categorised_data.keys() if Categorised_data[var]['VB']=='Price']

                price_variables=price_variables+['Generate Base Price']
                base_price_var=st.selectbox('Select base price if present or select "Generate Base Price" to generate ',price_variables,index=1) 
                if base_price_var=='Generate Base Price':
                    non_promo_price=st.selectbox('Select non promotional price variable to use for Base Price  generation',[var for var in price_variables if var!='Generate Base Price'])
                    promotional_price=st.selectbox('Select  promotional price variable to use for Base Price  generation',[var for var in price_variables if var!= non_promo_price and var!= 'Generate Base Price'])
                    base_price, discount_raw_series, discount_final_series=calculate_discount(data[promotional_price], data[non_promo_price])
                    data['base_price']=base_price
                    data['discount_raw']=discount_raw_series
                    data['discount_final']=discount_final_series
                    Categorised_data['base_price']={'VB':'Price','MB':"Price",'BB':'Internal'}
                    Categorised_data['discount_raw']={'VB':'Promotion','MB':"Promotion",'BB':'Internal'}
                    Categorised_data['discount_final']={'VB':'Promotion','MB':"Promotion",'BB':'Internal'}
                    fig=create_dual_axis_line_chart(data[time_variable], data[promotional_price],  data[non_promo_price], data['base_price'], data['discount_final'])
                    st.plotly_chart(fig,use_container_width=True)
            
            if st.checkbox('View Changes'):
                
                st.markdown('#### Updates Summary')
                st.session_state.data_processed = True
                with open('old_Categorised_data.pkl', 'rb') as file:
                    old_Categorised_data = pickle.load(file)
                cat_summary=  pd.DataFrame({'Old Name':variables,
                            "New Name":renamed_variables.values(),
                            "Old Bucket":[old_Categorised_data[col]["VB"] for col in  renamed_variables.keys()],
                            'New Bucket':[Categorised_data[col]["VB"] for col in  renamed_variables.values()],
                            })
                st.dataframe(cat_summary,use_container_width=True)
                    
                data.rename(columns=renamed_variables,inplace=True)
                # Bucket_summary={'Variable_name':data.columns,"Bucket":[Categorised_data[i]['VB'] for i in data.columns]}
                # st.dataframe(Bucket_summary)
                

            if st.button('Update Changes'):
                
                with open('Categorised_data.pkl', 'wb') as file:
                    pickle.dump(Categorised_data, file)

                
                with open("bucket_data_VB.pkl", "wb") as f:
                        pickle.dump(bucket_data_VB, f)
                
                data.set_index(time_variable,inplace=True)
                data.drop([var for var in time_variables if var!=time_variable],axis=1,inplace=True) 

                with open("edited_dataframe.pkl", "wb") as f:
                        pickle.dump(data, f)    

                st.success('Data updated and saved')
                data.fillna(0,inplace=True)
                st.dataframe(data.head(2))