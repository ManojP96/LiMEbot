import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from Eda_functions import format_numbers
import numpy as np
import pickle
from st_aggrid import AgGrid
from st_aggrid import GridOptionsBuilder,GridUpdateMode
from utilities import set_header,load_local_css
from st_aggrid import GridOptionsBuilder
import time
import itertools
import statsmodels.api as sm
import numpy as np
import re
import itertools
from sklearn.metrics import mean_absolute_error, r2_score,mean_absolute_percentage_error  
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
st.set_option('deprecation.showPyplotGlobalUse', False)
from datetime import datetime
import seaborn as sns
from Data_prep_functions import *
from Chatbot1_input_data import *
#from Chatbot_document import *
from streamlit_float import *
from Chatbot_model_output import *
from chatbot_document_copy import * 


st.set_page_config(
  page_title="Model Tuning",
  page_icon=":shark:",
  layout="wide",
  initial_sidebar_state='collapsed'
)
load_local_css('styles.css')
set_header()


st.title('1. Model Tuning')
if 'input_selected' not in st.session_state:
        st.session_state['input_selected']=False

if "output_selected" not in st.session_state:
    st.session_state['output_selected']=False

if 'radio2' not in st.session_state:
    st.session_state['radio2']=None

cols1,cols2=st.columns([9,2])        
float_init()

with cols2:

    container = st.container()
    with container:
        if st.session_state['radio2']=='Input Data':
            chatbot_input_data(disable=st.session_state['disabled'])
        elif st.session_state['radio2']=="Model Output":
                chatbot_model_output(disable=st.session_state['disabled'])
        else:
            chatbot1('Model Tuning',disable=st.session_state['disabled']) 

        js = """
        // Scroll container to bottom on page load
        window.onload = function() {
        document.querySelector('.streamlit-container').scrollTop = -500; 
        }
        """

        st.markdown(f'<script>{js}</script>', unsafe_allow_html=True) 
        #radios = st.radio('Select:', ['Input Data', 'Output Data'], key='radio_selection', on_change=radio)
        # st.write(st.session_state['radio2'])
        container.float("bottom: 2rem;background-color: white;overflow-y: scroll;max-height: 1000px;border: 1px solid #add8e6;")
    
with cols1:    
  with open("filtered_variables.pkl", 'rb') as file:
      filtered_variables = pickle.load(file)

  with open('Categorised_data.pkl', 'rb') as file:
    Categorised_data = pickle.load(file)

  with open("target_column.pkl", 'rb') as file:
    target_column= pickle.load(file)

  with open("df.pkl", 'rb') as file:
    df= pickle.load(file)
    df.fillna(0,inplace=True)
  with open("best_models.pkl", 'rb') as file:
    model_dict= pickle.load(file)

  if 'selected_model' not in st.session_state:
    st.session_state['selected_model']=0

  #st.write(list(model_dict.keys()).index(st.session_state["selected_model"]))
  # st.write(list(model_dict.keys()))
  st.session_state["selected_model"]=st.selectbox('Select Model to apply flags',model_dict.keys())
  model =model_dict[st.session_state["selected_model"]]['Model_object']
  date=df.index
  # model=st.session_state['Model']
  #st.write(model)
  X =model_dict[st.session_state["selected_model"]]['X']
  features_set= model_dict[st.session_state["selected_model"]]['feature_set']

  col=st.columns(3)  
  min_date=min(date)
  max_date=max(date)
  #st.write(date)
  with col[0]:
    start_date=st.date_input('Select Start Date',min_date,min_value=min_date,max_value=max_date)
  with col[1]:
    end_date=st.date_input('Select End Date',max_date,min_value=min_date,max_value=max_date)
  with col[2]:
    repeat=st.selectbox('Repeat Annually',['Yes','No'])
  if repeat =='Yes':
      repeat=True
  else: 
      repeat=False
  X=sm.add_constant(X)
  #st.text(start_date)
  if 'Flags' not in st.session_state:
    st.session_state['Flags']={}

  met,line_values,fig_flag=plot_actual_vs_predicted(date, df[target_column], model.predict(X), model,flag=(start_date,end_date),repeat_all_years=repeat)
  st.plotly_chart(fig_flag,use_container_width=True)
  flag_name='f1'
  flag_name=st.text_input('Enter Flag Name')
  if st.button('Update flag'):
    st.session_state['Flags'][flag_name]=line_values
    st.success(f'{flag_name} stored')

  options=list(st.session_state['Flags'].keys())
  selected_options = []
  num_columns = 4
  num_rows = -(-len(options) // num_columns)  # Ceiling division to calculate rows

  # Create a grid of checkboxes
  st.header('Select Flags for Model')
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


  if st.button('Build model with flags'):
    st.header('2.1 Results Summary')
    date=list(df.index)
    df = df.reset_index(drop=True)
    X=df[features_set]
    ss = MinMaxScaler()
    X = pd.DataFrame(ss.fit_transform(X), columns=X.columns)
    X=sm.add_constant(X)
    for flag in selected_options:
      X[flag]=st.session_state['Flags'][flag]
    y=df[target_column]
    model = sm.OLS(y, X).fit()
    st.write(model.summary())

    st.header('2.2 Actual vs. Predicted Plot')
    metrics_table,line,actual_vs_predicted_plot=plot_actual_vs_predicted(date, df[target_column], model.predict(X), model)
    st.plotly_chart(actual_vs_predicted_plot,use_container_width=True)
    st.markdown('## 2.3 Residual Analysis')
    columns=st.columns(2)
    with columns[0]:
      fig=plot_residual_predicted(df[target_column],model.predict(X),df)
      st.plotly_chart(fig)
    
    with columns[1]:
      st.empty()
      fig = qqplot(df[target_column],model.predict(X))
      st.plotly_chart(fig)

    with columns[0]:
      fig=residual_distribution(df[target_column],model.predict(X))
      st.pyplot(fig)

  if st.checkbox('Use this model to build response curves',key='123'):

    raw_data=df[features_set]
    columns_raw=[re.split(r"(_lag|_adst)",col)[0] for col in raw_data.columns]
    raw_data.columns=columns_raw
    columns_media=[col for col in columns_raw if Categorised_data[col]['BB']=='Media']
    raw_data=raw_data[columns_media]
    # conv_dict=({col:0.007 for col in columns_media if })
    # st.dataframe(raw_data.head(2))
    # st.write(columns_raw)
    raw_data['Date']=list(df.index)
    #raw_data(drop=True,inplace=True)
    spends_var=[col for col in df.columns if "spends" in col.lower() and 'adst' not in col.lower() and 'lag' not in col.lower()]
    spends_df=df[spends_var]
    spends_df['Week']=list(df.index)
    
    #spends_df.reset_index(drop=True,inplace=True)
    # st.dataframe(raw_data.head(2))
    # st.dataframe(spends_df.head(2))
    j=0
    X1=X.copy()
    col=X1.columns
    for i in model.params.values:
        X1[col[j]]=X1.iloc[:,j]*i
        j+=1
    contribution_df=X1
    contribution_df['Date']=list(df.index)
    excel_file='Overview_data.xlsx'

    with pd.ExcelWriter(excel_file,engine='xlsxwriter') as writer:
      raw_data.to_excel(writer,sheet_name='RAW DATA MMM',index=False)
      spends_df.to_excel(writer,sheet_name='SPEND INPUT',index=False)
      contribution_df.to_excel(writer,sheet_name='CONTRIBUTION MMM') 
    # st.dataframe(X1.head(2))
    # st.write(sum(X1.sum()))
    # st.write(sum(model.predict(X)))