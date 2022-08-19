# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:38:02 2022

@author: Srijani
"""

# In[1]:


import pandas as pd
import pickle
import streamlit as st
from PIL import Image
import scorecardpy as sc
import math 
import numpy as np


# In[2]:


# # load saved model
#with open('xgb_pkl_1' , 'rb') as f:
#     xgb = pickle.load(f)

pickle_in=open("C:\\Users\\srija\\Fraud_project\\fraud_detection_xgb.pkl","rb")
xgb=pickle.load(pickle_in)
# Getting Model Features
score_feat_xgb = xgb.get_booster().feature_names


# In[3]:




predictor=['Provider', 'InscClaimAmtReimbursed_merged_sum','InscClaimAmtReimbursed_merged_max','InscClaimAmtReimbursed_merged_std',
       'InscClaimAmtReimbursed_in_sum','InscClaimAmtReimbursed_in_min','InscClaimAmtReimbursed_in_max',
       'InscClaimAmtReimbursed_out_std','InscClaimAmtReimbursed_out_mean','InscClaimAmtReimbursed_out_max',
       'Race_3_merged_sum',
       'Race_2_merged_sum','Race_5_merged_sum','AttendingPhysician_out_nunique','OperatingPhysician_out_nunique',
       'OperatingPhysician_in_nunique',
       'OtherPhysician_out_nunique','OtherPhysician_in_nunique',
       'ChronicCond_Cancer_merged_sum','ChronicCond_stroke_merged_sum','ChronicCond_rheumatoidarthritis_merged_sum',
       'ChronicCond_KidneyDisease_merged_sum','ChronicCond_ObstrPulmonary_merged_sum','ChronicCond_Alzheimer_merged_sum',
       'ClmDiagnosisCode_5_in_count','ClmDiagnosisCode_6_out_count', 
       'ClmDiagnosisCode_7_out_nunique','ClmDiagnosisCode_7_in_nunique', 'ClmDiagnosisCode_8_in_count',
       'ClmDiagnosisCode_9_out_count',
       'ClmDiagnosisCode_9_in_count', 'ClmDiagnosisCode_9_in_nunique',
        'Claim_duration_merged_mean',
        'Age_merged_max','Age_merged_min','Age_merged_mean']




# In[4]:


def scoring(val_score):
    val_score['derived_ClmDiagnosisCode_9_flag'] = np.where(val_score['ClmDiagnosisCode_9_in_count']>val_score['ClmDiagnosisCode_9_in_nunique'], 1, 0)
    val_score['derived_Race_3_flag'] = np.where(val_score['Race_3_merged_sum']>=30, 1, 0)
    
    # now to apply woebins on validation sample
    # Read the dev_woe_validation dataframe
    dev_woe_validation=pd.read_csv("C:\\Users\\Sahil\\Desktop\\Summer\\Project\\dev_woe_validation.csv")
    val_score_woe = sc.woebin_ply(val_score[predictor], dev_woe_validation)
    
    woe_feat_val = [i for i in val_score_woe.columns if i.endswith('_woe') or i in (['Provider']) ]
    
    
    score_data=pd.merge(val_score_woe[woe_feat_val],val_score[['Provider','derived_ClmDiagnosisCode_9_flag','derived_Race_3_flag']]
                          , on='Provider', how='left')
    

    return(score_data)


# In[5]:


def predict_default(df):
    
    score_data_1=scoring(df)
    prediction=xgb.predict((score_data_1[score_feat_xgb]))
    return prediction


# In[6]:


def main():
    st.title("Healthcare Fraud Detection")
    html_temp = """
    <div style="background-color:orange;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Healthcare Fraud Detection ML App </h2>
    </div>
    """
    st.header('This app is created to predict if the Provider is fraud or not')
    st.markdown(html_temp,unsafe_allow_html=True)
    input_data=pd.DataFrame()
    for i in predictor:
        input_data[i] = st.text_input(i,"Type Here")
        
    #result=""

    if st.button("Predict default by XGBoost"):
        result=predict_default(input_data)
        if result==0:
           st.success('The Provider is non-fradulent')
        else:
           st.success('The Provider is fradulent') 
    if st.button("About"):
        st.text("Built with Streamlit")
if __name__=='__main__':
    main()

