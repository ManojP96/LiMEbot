import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from Eda_functions import *
import numpy as np
import re
import pickle
#from pandas_profiling import ProfileReport
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import streamlit as st
import streamlit.components.v1 as components
import sweetviz as sv
from utilities import set_header,initialize_data,load_local_css
from st_aggrid import GridOptionsBuilder,GridUpdateMode
from st_aggrid import GridOptionsBuilder
from st_aggrid import AgGrid
from streamlit_float import *
from Chatbot1_input_data import *
#from Chatbot_document import *
from streamlit_float import *
from Chatbot_model_output import *
from chatbot_document_copy import chatbot1


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
st.set_page_config(
  page_title="Data Validation",
  page_icon=":shark:",
  layout="wide",
  initial_sidebar_state='collapsed'
)
load_local_css('styles.css')
set_header()
#preprocessing
with open('Categorised_data.pkl', 'rb') as file:
  Categorised_data = pickle.load(file)
with open("edited_dataframe.pkl", 'rb') as file:
  df = pickle.load(file)
date=df.index
df.reset_index(inplace=True)
df['Date'] = pd.to_datetime(date)

#prospects=pd.read_excel('EDA_Data.xlsx',sheet_name='Prospects')
#spends=pd.read_excel('EDA_Data.xlsx',sheet_name='SPEND INPUT')
#spends.columns=['Week','Streaming (Spends)','TV (Spends)','Search (Spends)','Digital (Spends)']
#df=pd.concat([df,spends],axis=1)

#df['Date'] =pd.to_datetime(df['Date']).dt.strftime('%m/%d/%Y')
#df['Prospects']=prospects['Prospects']
#df.drop(['Week'],axis=1,inplace=True)

#streamlit code

if 'input_selected' not in st.session_state:
        st.session_state['input_selected']=False

if "output_selected" not in st.session_state:
    st.session_state['output_selected']=False

if 'radio2' not in st.session_state:
    st.session_state['radio2']=None

cols1,cols2=st.columns([9,2])        
float_init()



with cols2:
    container = st.container(border=True)

    with container:
        if st.session_state['radio2']=='Input Data':
            chatbot_input_data(disable=st.session_state['disabled'])
        elif st.session_state['radio2']=="Model Output":
            chatbot_model_output(disable=st.session_state['disabled'])
        else:
            chatbot1('Data Validation',disable=st.session_state['disabled']) 
        
        st.markdown('<div class="floating"></div>', unsafe_allow_html=True)
with cols1:
    
    st.title('Data Validation and Insights')
    target_variables=[col for col in df.drop('Date',axis=1).columns]
    target_column = st.selectbox('Select the Target Feature/Dependent Variable (will be used in all charts as reference)', [col for col in target_variables if Categorised_data[col]['VB']=='Sales'])
    with open("target_column.pkl", "wb") as f:
        pickle.dump(target_column, f)
        
    #st.write(target_column)
    fig=line_plot_target(df, target=target_column, title=f'{target_column} Over Time')
    st.plotly_chart(fig, use_container_width=True)

    # desired_columns = set([col for col in df.columns if 'imp' in col.lower() or 'cli' in col.lower() or 'spend' in col.lower()])
    # # automate?
    # desired_columns=list(desired_columns)
    # desired_columns.append(target_column)

    with open('Categorised_data.pkl', 'rb') as file:
        Categorised_data = pickle.load(file)
    #st.write(Categorised_data)

    #media

    media_channel=[col for col in Categorised_data.keys() if Categorised_data[col]['BB'] == "Media" ]
    # st.write(media_channel)

    Non_media_channel=[col for col in df.columns if col not in media_channel]


    st.markdown('### Annual Data Summary')
    st.dataframe(summary(df, media_channel+[target_column], spends=None,Target=True), use_container_width=True)

    if st.checkbox('Show raw data'):
        st.write(pd.concat([pd.to_datetime(df['Date']).dt.strftime('%m/%d/%Y'),df.select_dtypes(np.number).applymap(format_numbers)],axis=1))
    col1 = st.columns(1)

    if "selected_feature" not in st.session_state:
        st.session_state['selected_feature']=None

    st.header('1. Media Channels')

    if 'Validation' not in st.session_state:
        st.session_state['Validation']=[]

    selected_media = st.selectbox('Select media', np.unique([Categorised_data[col]['VB'] for col in media_channel]))
    # selected_feature=st.multiselect('Select Metric', df.columns[df.columns.str.contains(selected_media,case=False)])
    st.session_state["selected_feature"]=st.selectbox('Select Metric',[col for col in  media_channel  if    Categorised_data[col]['VB'] in selected_media ] )
    spends_features=[col for col in df.columns if 'spends' in col.lower() or 'cost' in col.lower()]
    spends_feature=[col for col in spends_features if col.split('_')[0] in st.session_state["selected_feature"].split('_')[0]]
    #st.write(spends_features)
    #st.write(spends_feature)
    #st.write(selected_feature)


    val_variables=[col for col in media_channel if col!='Date']
    if len(spends_feature)==0:  
        st.warning('No spends varaible available for the selected metric in data') 
        
    else:
        st.write(f'Selected spends variable {spends_feature[0]} if wrong please name the varaibles properly')
        # Create the dual-axis line plot
        fig_row1 = line_plot(df, x_col='Date', y1_cols=[st.session_state["selected_feature"]], y2_cols=[target_column], title=f'Analysis of {st.session_state["selected_feature"]} and {[target_column][0]} Over Time')
        st.plotly_chart(fig_row1, use_container_width=True)
        st.markdown('### Annual Data Summary')
        st.dataframe(summary(df,[st.session_state["selected_feature"]],spends=spends_feature[0]),use_container_width=True)
        if st.button('Validate'):
            st.session_state['Validation'].append(st.session_state["selected_feature"])

        if st.checkbox('Validate all'):
            st.session_state['Validation'].extend(val_variables)
            st.success('All media variables are validated ✅')
        if len(set(st.session_state['Validation']).intersection(val_variables))!=len(val_variables):
            #st.write(st.session_state['Validation'])
            validation_data=pd.DataFrame({'Variables':val_variables,
                                        'Validated':[1 if col in st.session_state['Validation'] else 0 for col in val_variables],
                                        'Bucket':[Categorised_data[col]['VB'] for col in val_variables]})
            gd=GridOptionsBuilder.from_dataframe(validation_data)
            gd.configure_pagination(enabled=True)
            gd.configure_selection(use_checkbox=True,selection_mode='multiple')
            #gd.configure_selection_toggle_all(None, show_toggle_all=True)
            #gd.configure_columns_auto_size_mode(GridOptionsBuilder.configure_columns)
            gridoptions=gd.build()
            #st.text(st.session_state['Validation'])
            table = AgGrid(validation_data,gridOptions=gridoptions,update_mode=GridUpdateMode.SELECTION_CHANGED,fit_columns_on_grid_load=True)
            #st.table(table)
            selected_rows = table["selected_rows"]
            st.session_state['Validation'].extend([col['Variables'] for col in selected_rows])
            not_validated_variables = [col for col in val_variables if col not in st.session_state["Validation"]]
            if not_validated_variables:
                not_validated_message = f'The following variables are not validated:\n{" , ".join(not_validated_variables)}'
                st.warning(not_validated_message)



    st.header('2. Non Media Variables')
    selected_columns_row = [col for col in df.columns if ("imp" not in col.lower()) and ('cli' not in col.lower() ) and ('spend' not in col.lower()) and col!='Date']
    selected_columns_row4 = st.selectbox('Select Channel',selected_columns_row )
    if not selected_columns_row4: 
        st.warning('Please select at least one.')
    else:
        # Create the dual-axis line plot
        fig_row4 = line_plot(df, x_col='Date', y1_cols=[selected_columns_row4], y2_cols=[target_column], title=f'Analysis of {selected_columns_row4} and {target_column} Over Time')
        st.plotly_chart(fig_row4, use_container_width=True)
        selected_non_media=selected_columns_row4
        sum_df = df[['Date', selected_non_media,target_column]]
        sum_df['Year']=pd.to_datetime(df['Date']).dt.year
        #st.dataframe(df)
        #st.dataframe(sum_df.head(2))
        sum_df=sum_df.groupby('Year').agg('sum')
        sum_df.loc['Grand Total']=sum_df.sum()         
        sum_df=sum_df.applymap(format_numbers) 
        sum_df.fillna('-',inplace=True)
        sum_df=sum_df.replace({"0.0":'-','nan':'-'})
        st.markdown('### Annual Data Summary')    
        st.dataframe(sum_df,use_container_width=True)

        # if st.checkbox('Validate',key='2'):
        #     st.session_state['Validation'].append(selected_columns_row4)
    # val_variables=[col for col in media_channel if col!='Date']
    # if st.checkbox('Validate all'):
    #     st.session_state['Validation'].extend(val_variables)
    # validation_data=pd.DataFrame({'Variables':val_variables,
    #                             'Validated':[1 if col in st.session_state['Validation'] else 0 for col in val_variables],
    #                             'Bucket':[Categorised_data[col]['VB'] for col in val_variables]})
    # gd=GridOptionsBuilder.from_dataframe(validation_data)
    # gd.configure_pagination(enabled=True)
    # gd.configure_selection(use_checkbox=True,selection_mode='multiple')
    # #gd.configure_selection_toggle_all(None, show_toggle_all=True)
    # #gd.configure_columns_auto_size_mode(GridOptionsBuilder.configure_columns)
    # gridoptions=gd.build()
    # #st.text(st.session_state['Validation'])
    # table = AgGrid(validation_data,gridOptions=gridoptions,update_mode=GridUpdateMode.SELECTION_CHANGED,fit_columns_on_grid_load=True)
    # #st.table(table)
    # selected_rows = table["selected_rows"]
    # st.session_state['Validation'].extend([col['Variables'] for col in selected_rows])
    # not_validated_variables = [col for col in val_variables if col not in st.session_state["Validation"]]
    # if not_validated_variables:
    #     not_validated_message = f'The following variables are not validated:\n{" , ".join(not_validated_variables)}'
    #     st.warning(not_validated_message)

    options = list(df.select_dtypes(np.number).columns)
    st.markdown(' ')
    st.markdown(' ')
    st.markdown('# Exploratory Data Analysis')
    st.markdown(' ')

    selected_options = []
    num_columns = 4
    num_rows = -(-len(options) // num_columns)  # Ceiling division to calculate rows

    # Create a grid of checkboxes
    st.header('Select Features for Correlation Plot')
    tick=False
    if st.checkbox('Select all'):
        tick=True
    selected_options = []
    for row in range(num_rows):
        cols = st.columns(num_columns)
        for col in cols:
            if options:
                option = options.pop(0) 
                selected = col.checkbox(option,value=tick)
                if selected:
                    selected_options.append(option)
    # Display selected options
    #st.write('You selected:', selected_options)
    st.pyplot(correlation_plot(df,selected_options,target_column))


    if st.button('Generate Profile Report'):
        pr = df.profile_report()

        st_profile_report(pr)

    if st.button('Generate Sweetviz Report'):
    
        def generate_report_with_target(df, target_feature):
            report = sv.analyze([df, "Dataset"], target_feat=target_feature)
            return report

        report = generate_report_with_target(df, target_feature=target_column)
        report.show_html()