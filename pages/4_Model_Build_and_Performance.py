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
from streamlit_float import *
from Chatbot1_input_data import *
#from Chatbot_document import *
from streamlit_float import *
from Chatbot_model_output import *
from chatbot_document_copy import * 

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
  page_title="Model Build",
  page_icon=":shark:",
  layout="wide",
  initial_sidebar_state='collapsed'
)
load_local_css('styles.css')
set_header()


st.title('1. Model Build')

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
            chatbot1('Model Build and Performance',st.session_state['disabled']) 
        
        st.markdown('<div class="floating"></div>', unsafe_allow_html=True)
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

  if "selected_rows" not in st.session_state:
    st.session_state['selected_rows']=None

  if 'final_selection' not in st.session_state:
      st.session_state['final_selection']=None

  #st.write(filtered_variables.keys()) #find

  media_channels=[col for col in filtered_variables.keys() if Categorised_data[col]['BB']=='Media']
  unique_media_variables=list(np.unique([Categorised_data[col]['VB'] for col in media_channels]))


  for i in unique_media_variables:
    filtered_variables[i]=[]
    for j in media_channels:
        if Categorised_data[j]['VB']==i:
          filtered_variables[i]=filtered_variables[i]+filtered_variables[j]
          del filtered_variables[j]


  if st.button('Create all possible combinations of variables'):
    with st.spinner('Wait for it'):
      multiple_col=[col for col in [i for i in filtered_variables.keys() if i not in unique_media_variables] if Categorised_data[col]['VB']=='Holiday']
      #st.write(multiple_col)


      for var in multiple_col:  
        all_combinations_hol = []
        for r in range(1, len(filtered_variables[var]) + 1):
            combinations = itertools.combinations(filtered_variables[var], r)
            all_combinations_hol.extend(combinations)
        all_combinations_hol.append([])
        all_combinations_hol = [list(comb) for comb in all_combinations_hol] 
        filtered_variables[var]=all_combinations_hol


    
      price=[col for col in df.columns if Categorised_data[re.split(r'_adst|_lag', col )[0]]['VB']=='Price']
      
      
      #st.write(price)
      Distribution=[col for col in df.columns if Categorised_data[re.split(r'_adst|_lag', col )[0]]['VB']=='Distribution']
      Promotion=[col for col in df.columns if  Categorised_data[re.split(r'_adst|_lag', col )[0]]['VB']=='Promotion']
      
      price.append('')
      Distribution.append('')
      

      filtered_variables['Price']=price
      filtered_variables['Distribution']=Distribution
      filtered_variables['Promotion']=Promotion

      variable_names = list(filtered_variables.keys())
      variable_values = list(filtered_variables.values())

      combinations = list(itertools.product(*variable_values))

      final_selection=[]
      for comb in combinations:
        nested_tuple = comb

        flattened_list = [item for sublist in nested_tuple for item in (sublist if isinstance(sublist, list) else [sublist])]
        final_selection.append(flattened_list)
      #st.write(final_selection[:15])

      st.session_state['final_selection']=final_selection

      st.success('Done')

  if 'Model_results' not in st.session_state:
        st.session_state['Model_results']={'Model_object':[],
      'Model_iteration':[],
      'Feature_set':[],
      'MAPE':[],
      'R2':[],
      'ADJR2':[]
      }

  #if st.button('Build Model'):
  if 'iterations' not in st.session_state:
    st.session_state['iterations']=1
  save_path = "Model"
  if st.session_state["final_selection"] is not None:
    st.write(f'Total combinations created {format_numbers(len(st.session_state["final_selection"]))}')
  if st.checkbox('Build all iterations'):
    iterations=len(st.session_state['final_selection'])
  else:
    iterations = st.number_input('Select the number of iterations to perform', min_value=1, step=100, value=st.session_state['iterations'])  

  st.session_state['iterations']=iterations

  if st.button("Build Models"):
    
    progress_bar = st.progress(0)  # Initialize the progress bar
    #time_remaining_text = st.empty()  # Create an empty space for time remaining text
    start_time = time.time()  # Record the start time
    progress_text = st.empty()
    #time_elapsed_text = st.empty()

    for i, selected_features in enumerate(st.session_state["final_selection"][:int(iterations)]):
        df = df.reset_index(drop=True)

        fet = [var for var in selected_features if len(var) > 0]
        X = df[fet]
        y = df['Prospects']
        ss = MinMaxScaler()
        X = pd.DataFrame(ss.fit_transform(X), columns=X.columns)
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        # st.write(fet)
        positive_coeff=[col for col in fet if Categorised_data[re.split(r'_adst|_lag', col )[0]]['VB'] in ["Distribution","Promotion	TV"	,"Display",	"Video"	,"Facebook",	"Twitter"	,"Instagram"	,"Pintrest",	"YouTube"	,"Paid Search"	,"OOH	Radio"	,"Audio Streaming",'Digital']]  
        negetive_coeff=[col for col in fet  if Categorised_data[re.split(r'_adst|_lag', col )[0]]['VB'] in ["Price"]]
        coefficients=model.params.to_dict()
        model_possitive=[col for col in coefficients.keys() if coefficients[col]>0]
        model_negatives=[col for col in coefficients.keys() if coefficients[col]<0]
        # st.write(positive_coeff)
        # st.write(model_possitive)
        pvalues=[var for var in list(model.pvalues) if var<=0.06]
        if (set(positive_coeff).issubset(set(model_possitive))) and (set(negetive_coeff).issubset(model_negatives)) and (len(pvalues)/len(selected_features))>=0.5:


            predicted_values = model.predict(X)
            mape = mean_absolute_percentage_error(y, predicted_values)
            adjr2 = model.rsquared_adj
            r2 = model.rsquared
            filename = os.path.join(save_path, f"model_{i}.pkl")
            with open(filename, "wb") as f:
              pickle.dump(model, f)
            # with open(r"C:\Users\ManojP\Documents\MMM\simopt\Model\model.pkl", 'rb') as file:
            #   model = pickle.load(file)

            st.session_state['Model_results']['Model_object'].append(filename)
            st.session_state['Model_results']['Model_iteration'].append(i)
            st.session_state['Model_results']['Feature_set'].append(fet)
            st.session_state['Model_results']['MAPE'].append(mape)
            st.session_state['Model_results']['R2'].append(r2)
            st.session_state['Model_results']['ADJR2'].append(adjr2)

        current_time = time.time()
        time_taken = current_time - start_time
        time_elapsed_minutes = time_taken / 60
        completed_iterations_text = f"{i + 1}/{iterations}"
        progress_bar.progress((i + 1) / int(iterations))
        progress_text.text(f'Completed iterations: {completed_iterations_text},Time Elapsed (min): {time_elapsed_minutes:.2f}')
    
    #pd.DataFrame(st.session_state['Model_results']).to_csv('model_result_data.csv',index=False)
  st.write(f'Out of {st.session_state["iterations"]} iterations : {len(st.session_state["Model_results"]["Model_object"])} valid models')

  if len(st.session_state['Model_results']['Model_object'])>0:
     st.session_state['disabled']=False
    #  st.rerun()



  def to_percentage(value):
    return f'{value * 100:.1f}%'   

  st.title('2. Select Models')
  if 'tick' not in st.session_state:
    st.session_state['tick']=False
  if st.checkbox('Show results of top 10 models (based on MAPE and Adj. R2)',value=st.session_state['tick']):
    st.session_state['tick']=True
    st.write('Select one model iteration to generate performance metrics for it:')
    data=pd.DataFrame(st.session_state['Model_results'])
    data.sort_values(by=['MAPE'],ascending=False,inplace=True)
    data.drop_duplicates(subset='Model_iteration',inplace=True)
    top_10=data.head(10)
    top_10['Rank']=np.arange(1,len(top_10)+1,1)
    top_10[['MAPE','R2','ADJR2']]=np.round(top_10[['MAPE','R2','ADJR2']],4).applymap(to_percentage)
    top_10_table = top_10[['Rank','Model_iteration','MAPE','ADJR2','R2']]
    #top_10_table.columns=[['Rank','Model Iteration Index','MAPE','Adjusted R2','R2']]
    gd=GridOptionsBuilder.from_dataframe(top_10_table)
    gd.configure_pagination(enabled=True)
    gd.configure_selection(use_checkbox=True)

    
    gridoptions=gd.build()

    table = AgGrid(top_10,gridOptions=gridoptions,update_mode=GridUpdateMode.SELECTION_CHANGED)
    
    selected_rows=table.selected_rows
    if st.session_state["selected_rows"] != selected_rows:
      st.session_state["build_rc_cb"] = False
    st.session_state["selected_rows"] = selected_rows
    if 'Model' not in st.session_state:
      st.session_state['Model']={}

    # mod=st.button('Use this model to add flags',key='mod')
    # if mod:
    #     mod_name=st.text_input('Enter model name')
    #     if len(mod_name)>0:
    #       st.success('Model saved move to next page to tune!')
    #       st.session_state['Model'][mod_name]={"Model_object":model,'feature_set':st.session_state['features_set'],'X':st.session_state['X']}
      
    if len(selected_rows)>0:
      st.header('2.1 Results Summary')

      model_object=data[data['Model_iteration']==selected_rows[0]['Model_iteration']]['Model_object']
      features_set=data[data['Model_iteration']==selected_rows[0]['Model_iteration']]['Feature_set']
      
      with open(str(model_object.values[0]), 'rb') as file:
              model = pickle.load(file)
      st.write(model.summary())        
      st.header('2.2 Actual vs. Predicted Plot')
      

      date=list(df.index)
      df = df.reset_index(drop=True)
      X=df[features_set.values[0]]
      ss = MinMaxScaler()
      X = pd.DataFrame(ss.fit_transform(X), columns=X.columns)
      X=sm.add_constant(X)
      st.session_state['X']=X
      st.session_state['features_set']=features_set.values[0]
      metrics_table,line,actual_vs_predicted_plot=plot_actual_vs_predicted(date, df[target_column], model.predict(X), model,target_column=target_column)

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
      


      vif_data = pd.DataFrame()
      # X=X.drop('const',axis=1)
      vif_data["Variable"] = X.columns
      vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
      vif_data.sort_values(by=['VIF'],ascending=False,inplace=True)
      vif_data=np.round(vif_data)
      vif_data['VIF']=vif_data['VIF'].astype(int)
      st.header('2.4 Variance Inflation Factor (VIF)')
      #st.dataframe(vif_data)
      color_mapping = {
      'darkgreen': (vif_data['VIF'] < 3),
      'orange': (vif_data['VIF'] >= 3) & (vif_data['VIF'] <= 10),
      'darkred': (vif_data['VIF'] > 10)
      }

  # Create a horizontal bar plot
      fig, ax = plt.subplots()
      fig.set_figwidth(10)  # Adjust the width of the figure as needed

      # Sort the bars by descending VIF values
      vif_data = vif_data.sort_values(by='VIF', ascending=False)

      # Iterate through the color mapping and plot bars with corresponding colors
      for color, condition in color_mapping.items():
          subset = vif_data[condition]
          bars = ax.barh(subset["Variable"], subset["VIF"], color=color, label=color)
          
          # Add text annotations on top of the bars
          for bar in bars:
              width = bar.get_width()
              ax.annotate(f'{width:}', xy=(width, bar.get_y() + bar.get_height() / 2), xytext=(5, 0),
                          textcoords='offset points', va='center')

      # Customize the plot
      ax.set_xlabel('VIF Values')
      #ax.set_title('2.4 Variance Inflation Factor (VIF)')
      #ax.legend(loc='upper right')

      # Display the plot in Streamlit
      st.pyplot(fig)
      value=False
      if st.checkbox('Use this model to build response curves',key='build_rc_cb'):
        mod_name=st.text_input('Enter model name')
        if len(mod_name)>0:
          st.session_state['Model'][mod_name]={"Model_object":model,'feature_set':st.session_state['features_set'],'X':st.session_state['X']}
        
          with open("best_models.pkl", "wb") as f:
            pickle.dump(st.session_state['Model'], f)  
            st.success('Model saved!, Proceed  next page to tune')
          value=False



      #st.write(Categorised_data.keys())
      #for response curve
      # raw_data_col=[col for col in X.columns if 'imp' in col or  'cli' in col or 'spend' in col]
      # raw_data_col_up=[re.split(r'(_adst|_lag)',col)[0] for col in raw_data_col ]
      # raw_data=X[raw_data_col]
      # raw_data.columns=raw_data_col_up

      #raw_data.to_excel('raw_data.xlsx',index=False)

      #t.dataframe(raw_data)

      # st.markdown('')
      # st.markdown('')
      # st.markdown('## 3. Add Events (Flags)')
      # col=st.columns(3)  
      # min_date=min(date)
      # max_date=max(date)
      # #st.write(date)
      # with col[0]:
      #   start_date=st.date_input('Select Start Date',min_date,min_value=min_date,max_value=max_date)
      # with col[1]:
      #   end_date=st.date_input('Select End Date',max_date,min_value=min_date,max_value=max_date)
      # with col[2]:
      #   repeat=st.selectbox('Repeat Annually',['Yes','No'])
      # if repeat =='Yes':
      #     repeat=True
      # else: 
      #     repeat=False
      # X=sm.add_constant(X)
      # #st.text(start_date)
      # met,line_values,fig_flag=plot_actual_vs_predicted(date, df[target_column], model.predict(X), model,flag=(start_date,end_date),repeat_all_years=repeat)
      # st.plotly_chart(fig_flag,use_container_width=True)
      # flag_name='f1'
      # flag_name=st.text_input('Enter Flag Name')
      # if st.button('Update flag to model'):
      #   st.header('2.1 Results Summary')
      #   date=list(df.index)
      #   df = df.reset_index(drop=True)
      #   X=df[features_set.values[0]]
      #   ss = MinMaxScaler()
      #   X = pd.DataFrame(ss.fit_transform(X), columns=X.columns)
      #   X=sm.add_constant(X)
      #   X[flag_name]=line_values
      #   y=df[target_column]
      #   model = sm.OLS(y, X).fit()
      #   st.write(model.summary())

      #   st.header('2.2 Actual vs. Predicted Plot')
      #   metrics_table,line,actual_vs_predicted_plot=plot_actual_vs_predicted(date, df[target_column], model.predict(X), model)
      #   st.plotly_chart(actual_vs_predicted_plot,use_container_width=True)
      #   st.markdown('## 2.3 Residual Analysis')
      #   columns=st.columns(2)
      #   with columns[0]:
      #     fig=plot_residual_predicted(df[target_column],model.predict(X),df)
      #     st.plotly_chart(fig)
        
      #   with columns[1]:
      #     st.empty()
      #     fig = qqplot(df[target_column],model.predict(X))
      #     st.plotly_chart(fig)

      #   with columns[0]:
      #     fig=residual_distribution(df[target_column],model.predict(X))
      #     st.pyplot(fig)

      # st.checkbox('Use this model to build response curves',key='123')
