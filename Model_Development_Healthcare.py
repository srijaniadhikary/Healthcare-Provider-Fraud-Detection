#!/usr/bin/env python
# coding: utf-8

# # In this Notebook we will do Model Development part.
# ## For Model Development we have to do various steps before that part like-Development and Validation sample, Data preparation, EDA,Feature selection,Model development. After the model development we will do scoring part. 
# ### In the Previous Notebook we have prepared our data on which we will develop and score a model and check its accuracy. Now we will do the Development part.

# Importing the Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot 
from numpy import sort
import seaborn as sns
import os
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# ### Loading the datasets of train_provider and health. The health dataset we have already got from the previous notebook.

# In[2]:


train_provider=pd.read_csv("C:\\Users\\Sahil\\Desktop\\Summer\\Project\\Dataset\\Train-1542865627584.csv")
health=pd.read_csv('C:\\Users\\Sahil\\Desktop\\Summer\\Project\\df_healthcare.csv')


# In[3]:


pd.set_option('display.max_columns', None)


# In[4]:


health.shape


# - In the dataset health for some features there is no description available about them. We wanted to avoid any kind of  misinterpretaion caused by those features. So we dropped them. first made a list as drop_feat that contains a list of features that we should drop and then we droped them from the health dataset. The dropped features are (IPAnnualReimbursementAmt, IPAnnualDeductibleAmt, OPAnnualReimbursementAmt, OPAnnualDeductibleAmt and their mean, min,max,std) 

# In[5]:


drop_feat = [i for i in health.columns if i.startswith('OPA') or i.startswith('IPA')]


# In[6]:


drop_feat


# In[7]:


health=health.drop(drop_feat,axis=1)


# In[8]:


health.head()


# In[9]:


health.shape


# - Now the shape of the health dataset is (5410,84). Total number of provers is 5410 and total number of features is 83 and one column is of provider. 
# - We set the values of PotentialFraud (Yes/No) to (1/0) in the train_provider dataset.

# In[10]:


train_provider=train_provider.replace({'PotentialFraud':'Yes'},1)
train_provider=train_provider.replace({'PotentialFraud':'No'},0)


# In[11]:


train_provider.shape


# In[12]:


train_provider.head()


# ## Now split the train_provider into two parts:
# - one for model development part(data1) and other for validation part(data2) and save them as csv files for future references.
# <br>
# 
# **Development Sample**:
# - This is the sample on which we'll prepare the Model
# - All EDA, Pre-processing, Feature Engineering will be done on this sample <br>
# 
# **Validation Sample**
# - This is the sample on which the model will be scored
# - All features used in the model will be created separately in this data
# - Only features that are selected in the final model will be derived from this sample <br>
# 
# data1 contains 5004 rows and data2 contains 406 rows. 

# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


data1,data2=train_test_split(train_provider,test_size=0.075,random_state=0,stratify = train_provider['PotentialFraud'])


# In[15]:


data1.shape


# In[16]:


data2.shape


# In[23]:


data1.head()


# In[22]:


data2.head()


# In[4]:


print(data1['PotentialFraud'].value_counts())
print(data2['PotentialFraud'].value_counts())


# In[5]:


data1.groupby('PotentialFraud')['PotentialFraud'].count().plot(kind='pie',y='PotentialFraud',autopct='%1.1f%%')


# In[241]:


data1.to_csv('C:\\Users\\Sahil\\Desktop\\Summer\\Project\\data1.csv',index=False)


# In[2]:


data1=pd.read_csv('C:\\Users\\Sahil\\Desktop\\Summer\\Project\\data1.csv')


# In[248]:


data2.to_csv('C:\\Users\\Sahil\\Desktop\\Summer\\Project\\data2.csv',index=False)


# In[3]:


data_2=pd.read_csv('C:\\Users\\Sahil\\Desktop\\Summer\\Project\\data2.csv')
data2=data_2.drop(columns='Unnamed: 0')


# ## Merge the data1 set & and the data 2 set with health dataset on provider to make the development dataset & Validation set respectively. 
# We get health_development as development dataset and health_validation as validation dataset.

# In[27]:


health_development=pd.merge(health,data1, on='Provider')
health_validation=pd.merge(health,data2, on='Provider')


# In[28]:


print(health_development.shape)
print(health_validation.shape)


# Now each dataset contains 85 columns among which 83 are features and 2 are Provider and PotentialFraud

# In[26]:


health_validation.head()


# - Here we will work on development data to make our model. we will not touch the validation dataset. we will save that file in a csv file named  health_validation on which we will perform our model scoring in the scoring part

# In[23]:


health_validation.to_csv('C:\\Users\\Sahil\\Desktop\\Summer\\Project\\health_validation.csv',index=False)


# In[29]:


dev=health_development


# In[30]:


dev.head()


# # Data Preparation : EDA, Preprocessing and Feature Engineering on the Development sample

# ### Create user derived Features
# - Use a prefix on each derived feature like "derived" in order to quickly filter feature names

# In[33]:


dev_copy = dev.copy(deep=True)


# 1. Deriving a flag from the distribution of Age_merged_max column

# In[34]:


fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["Age_merged_max"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["Age_merged_max"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of Age_merged_max', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()


# In[35]:


x=dev_copy[dev_copy['Age_merged_max'] >100]


# In[36]:


x.head(2)


# In[41]:


print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))


# so we created a flag called derived_age_max_flag where 1 denotes Age_merged_max for that Provider is >100

# In[42]:


dev_copy['derived_age_max_flag'] = np.where(dev_copy['Age_merged_max'] >100 , 1, 0)


# 2. Deriving a flag from the distribution of Age_merged_min column

# In[43]:


x=dev_copy[dev_copy['Age_merged_min'] <30]


# In[44]:


print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))


# So we created a flag called derived_age_min_flag where 1 denotes Age_merged_min for that Provider is <30

# In[46]:


dev_copy['derived_age_min_flag'] = np.where(dev_copy['Age_merged_min']<30  , 1, 0)


# 3. Deriving a flag from the  InscClaimAmtReimbursed_merged_max column
# 4. Deriving a flag from the  InscClaimAmtReimbursed_in_max column <br>
# 
# Both the flags are based on mean+3*std deviation as 1 whether the max amount is greater or equal to mean+3*std deviation

# In[47]:


#mean+3sigma flag of InscClaimAmtReimbursed for inpatient and outpatient also
dev_copy['derived_InscClaimAmtReimbursed_merged_flag'] = np.where(dev_copy['InscClaimAmtReimbursed_merged_max']>= dev_copy['InscClaimAmtReimbursed_merged_mean']+3*dev_copy['InscClaimAmtReimbursed_merged_std'] , 1, 0)

dev_copy['derived_InscClaimAmtReimbursed_in_flag'] = np.where(dev_copy['InscClaimAmtReimbursed_in_max']>= dev_copy['InscClaimAmtReimbursed_in_mean']+3*dev_copy['InscClaimAmtReimbursed_in_std'] , 1, 0)


# - **Now we are going to derive flags on the basis of distribution of various diseases** <br>
# The reasons are stated below :

# In[62]:


#from Diseases ChronicCond_Alzheimer_merged_sum here in range (100,500) with event rate almost 50% 

dev_copy['derived_ChronicCond_Alzheimer_flag'] = np.where(dev_copy['ChronicCond_Alzheimer_merged_sum']>=150 , 1, 0)

#from Diseases ChronicCond_Heartfailure_merged_sum here in range (150,20) with event rate 32.8% 

dev_copy['derived_ChronicCond_Heartfailure_flag'] = np.where(dev_copy['ChronicCond_Heartfailure_merged_sum']>=150, 1, 0)

#from Diseases ChronicCond_KidneyDisease_merged_sum here in range (100,2000) with event rate % 

dev_copy['derived_ChronicCond_KidneyDisease_flag'] = np.where(dev_copy['ChronicCond_KidneyDisease_merged_sum']>=150, 1, 0)

dev_copy['derived_ChronicCond_Cancer_flag'] = np.where(dev_copy['ChronicCond_Cancer_merged_sum']>=70, 1, 0)

dev_copy['derived_ChronicCond_ObstrPulmonary_flag'] = np.where(dev_copy['ChronicCond_ObstrPulmonary_merged_sum']>=150 , 1, 0)

dev_copy['derived_ChronicCond_Depression_flag'] = np.where(dev_copy['ChronicCond_Depression_merged_sum']>=200 , 1, 0)

dev_copy['derived_ChronicCond_Diabetes_flag'] = np.where(dev_copy['ChronicCond_Diabetes_merged_sum']>=250, 1, 0)

dev_copy['derived_ChronicCond_IschemicHeart_flag'] = np.where(dev_copy['ChronicCond_IschemicHeart_merged_sum']>=350, 1, 0)

dev_copy['derived_ChronicCond_Osteoporasis_flag'] = np.where(dev_copy['ChronicCond_Osteoporasis_merged_sum']>=150, 1, 0)

dev_copy['derived_ChronicCond_rheumatoidarthritis_flag'] = np.where(dev_copy['ChronicCond_rheumatoidarthritis_merged_sum']>=150, 1, 0)

dev_copy['derived_ChronicCond_stroke_flag'] = np.where(dev_copy['ChronicCond_stroke_merged_sum']>=50, 1, 0)


# In[51]:


#derived_ChronicCond_Alzheimer_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["ChronicCond_Alzheimer_merged_sum"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["ChronicCond_Alzheimer_merged_sum"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of ChronicCond_Alzheimer_merged_sum', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['ChronicCond_Alzheimer_merged_sum'] >= 150]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if ChronicCond_Alzheimer_merged_sum >=150 otherwise 0")


# In[53]:


#derived_ChronicCond_Heartfailure_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["ChronicCond_Heartfailure_merged_sum"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["ChronicCond_Heartfailure_merged_sum"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of ChronicCond_Heartfailure_merged_sum', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['ChronicCond_Heartfailure_merged_sum'] >= 150]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if ChronicCond_Heartfailure_merged_sum >=150 otherwise 0")


# In[52]:


#derived_ChronicCond_KidneyDisease_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["ChronicCond_KidneyDisease_merged_sum"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["ChronicCond_KidneyDisease_merged_sum"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of ChronicCond_KidneyDisease_merged_sum', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['ChronicCond_KidneyDisease_merged_sum'] >= 150]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if ChronicCond_KidneyDisease_merged_sum >=150 otherwise 0")


# In[54]:


#derived_ChronicCond_Cancer_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["ChronicCond_Cancer_merged_sum"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["ChronicCond_Cancer_merged_sum"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of ChronicCond_Cancer_merged_sum', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['ChronicCond_Cancer_merged_sum'] >= 70]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if ChronicCond_Cancer_merged_sum >=70 otherwise 0")


# In[55]:


#derived_ChronicCond_ObstrPulmonary_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["ChronicCond_ObstrPulmonary_merged_sum"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["ChronicCond_ObstrPulmonary_merged_sum"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of ChronicCond_ObstrPulmonary_merged_sum', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['ChronicCond_ObstrPulmonary_merged_sum'] >= 150]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if ChronicCond_ObstrPulmonary_merged_sum >=150 otherwise 0")


# In[56]:


#derived_ChronicCond_Depression_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["ChronicCond_Depression_merged_sum"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["ChronicCond_Depression_merged_sum"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of ChronicCond_Depression_merged_sum', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['ChronicCond_Depression_merged_sum'] >= 200]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if ChronicCond_Depression_merged_sum >=200 otherwise 0")


# In[57]:


#derived_ChronicCond_Diabetes_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["ChronicCond_Diabetes_merged_sum"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["ChronicCond_Diabetes_merged_sum"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of ChronicCond_Diabetes_merged_sum', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['ChronicCond_Diabetes_merged_sum'] >= 250]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if ChronicCond_Diabetes_merged_sum >=250 otherwise 0")


# In[58]:


#derived_ChronicCond_IschemicHeart_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["ChronicCond_IschemicHeart_merged_sum"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["ChronicCond_IschemicHeart_merged_sum"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of ChronicCond_IschemicHeart_merged_sum', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['ChronicCond_IschemicHeart_merged_sum'] >= 350]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if ChronicCond_IschemicHeart_merged_sum >=350 otherwise 0")


# In[59]:


#derived_ChronicCond_Osteoporasis_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["ChronicCond_Osteoporasis_merged_sum"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["ChronicCond_Osteoporasis_merged_sum"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of ChronicCond_Osteoporasis_merged_sum', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['ChronicCond_Osteoporasis_merged_sum'] >= 150]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if ChronicCond_Osteoporasis_merged_sum >=150 otherwise 0")


# In[60]:


#derived_ChronicCond_rheumatoidarthritis_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["ChronicCond_rheumatoidarthritis_merged_sum"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["ChronicCond_rheumatoidarthritis_merged_sum"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of ChronicCond_rheumatoidarthritis_merged_sum', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['ChronicCond_rheumatoidarthritis_merged_sum'] >= 150]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if ChronicCond_rheumatoidarthritis_merged_sum >=150 otherwise 0")


# In[61]:


#derived_ChronicCond_stroke_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["ChronicCond_stroke_merged_sum"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["ChronicCond_stroke_merged_sum"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of ChronicCond_stroke_merged_sum', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['ChronicCond_stroke_merged_sum'] >= 50]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if ChronicCond_stroke_merged_sum >=50 otherwise 0")


# - *Now from Genders deriving flags*

# In[31]:


#from genders 

dev_copy['derived_Gender_1_flag'] = np.where(dev_copy['Gender_1_merged_sum']>200, 1, 0)
dev_copy['derived_Gender_2_flag'] = np.where(dev_copy['Gender_2_merged_sum']>250, 1, 0)


# In[64]:


#derived_Gender_1_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["Gender_1_merged_sum"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["Gender_1_merged_sum"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of Gender_1_merged_sum', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['Gender_1_merged_sum'] > 200]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if Gender_1_merged_sum >200 otherwise 0")


#derived_Gender_2_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["Gender_2_merged_sum"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["Gender_2_merged_sum"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of Gender_2_merged_sum', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['Gender_2_merged_sum'] > 250]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if ChronicCond_stroke_merged_sum >250 otherwise 0")


# - *Now from OperatingPhysician_out_nunique and OtherPhysician_out_nunique deriving flags*

# In[32]:


#from OperatingPhysician_out_nunique

dev_copy['derived_OperatingPhysician_flag'] = np.where(dev_copy['OperatingPhysician_out_nunique']>=30, 1, 0)
dev_copy['derived_OtherPhysician_flag'] = np.where(dev_copy['OtherPhysician_out_nunique'].between(60,500), 1, 0)


# In[65]:


#derived_OperatingPhysician_out_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["OperatingPhysician_out_nunique"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["OperatingPhysician_out_nunique"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of OperatingPhysician_out_nunique', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['OperatingPhysician_out_nunique'] >=30]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if OperatingPhysician_out_nunique >=30 otherwise 0")


#derived_OtherPhysician_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["OtherPhysician_out_nunique"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["OtherPhysician_out_nunique"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of OtherPhysician_out_nunique', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['OtherPhysician_out_nunique'].between(60,500)]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if OtherPhysician_out_nunique is in between 60 and 500 otherwise 0")


# - *From claim diagonosis codes we create flags where it takes value 1 if it's count is greater or equal to unique count*

# In[67]:


#from claim diagnosis codes ClmDiagnosisCodes


dev_copy['derived_ClmDiagnosisCode_1_flag'] = np.where(dev_copy['ClmDiagnosisCode_1_in_count']>dev_copy['ClmDiagnosisCode_1_in_nunique'], 1, 0)
dev_copy['derived_ClmDiagnosisCode_2_flag'] = np.where(dev_copy['ClmDiagnosisCode_2_in_count']>dev_copy['ClmDiagnosisCode_2_in_nunique'], 1, 0)
dev_copy['derived_ClmDiagnosisCode_3_flag'] = np.where(dev_copy['ClmDiagnosisCode_3_in_count']>dev_copy['ClmDiagnosisCode_3_in_nunique'], 1, 0)
dev_copy['derived_ClmDiagnosisCode_4_flag'] = np.where(dev_copy['ClmDiagnosisCode_4_in_count']>dev_copy['ClmDiagnosisCode_4_in_nunique'], 1, 0)
dev_copy['derived_ClmDiagnosisCode_5_flag'] = np.where(dev_copy['ClmDiagnosisCode_5_in_count']>dev_copy['ClmDiagnosisCode_5_in_nunique'], 1, 0)
dev_copy['derived_ClmDiagnosisCode_6_flag'] = np.where(dev_copy['ClmDiagnosisCode_6_in_count']>dev_copy['ClmDiagnosisCode_6_in_nunique'], 1, 0)
dev_copy['derived_ClmDiagnosisCode_7_flag'] = np.where(dev_copy['ClmDiagnosisCode_7_in_count']>dev_copy['ClmDiagnosisCode_7_in_nunique'], 1, 0)
dev_copy['derived_ClmDiagnosisCode_8_flag'] = np.where(dev_copy['ClmDiagnosisCode_8_in_count']>dev_copy['ClmDiagnosisCode_8_in_nunique'], 1, 0)
dev_copy['derived_ClmDiagnosisCode_9_flag'] = np.where(dev_copy['ClmDiagnosisCode_9_in_count']>dev_copy['ClmDiagnosisCode_9_in_nunique'], 1, 0)



# In[68]:


#from the races Race_1_merged_sum

dev_copy['derived_Race_1_flag'] = np.where(dev_copy['Race_1_merged_sum']>=np.percentile(dev_copy['Race_1_merged_sum'],95), 1, 0)
dev_copy['derived_Race_2_flag'] = np.where(dev_copy['Race_2_merged_sum']>=np.percentile(dev_copy['Race_2_merged_sum'],95), 1, 0)
dev_copy['derived_Race_3_flag'] = np.where(dev_copy['Race_3_merged_sum']>=30, 1, 0)
dev_copy['derived_Race_5_flag'] = np.where(dev_copy['Race_5_merged_sum']>=20, 1, 0)


# In[66]:


#derived_Race_1_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["Race_1_merged_sum"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["Race_1_merged_sum"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of Race_1_merged_sum', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['Race_1_merged_sum']>=np.percentile(dev_copy['Race_1_merged_sum'],95)]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if Race_1_merged_sum >= 95th percentile  otherwise 0")


#derived_Race_2_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["Race_2_merged_sum"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["Race_2_merged_sum"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of Race_2_merged_sum', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['Race_2_merged_sum']>=np.percentile(dev_copy['Race_2_merged_sum'],95)]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if Race_2_merged_sum 95th percentile otherwise 0")

#derived_Race_3_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["Race_3_merged_sum"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["Race_3_merged_sum"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of Race_3_merged_sum', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['Race_3_merged_sum'] >= 30]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if Race_3_merged_sum >=30 otherwise 0")


#derived_Race_5_flag

fig, ax = plt.subplots(figsize = (13,5))
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==1]["Race_5_merged_sum"], alpha=0.5,shade = True, color="red", label="PotentialFraud", ax = ax)
sns.kdeplot(dev_copy[dev_copy["PotentialFraud"]==0]["Race_5_merged_sum"], alpha=0.5,shade = True, color="#fccc79", label="Normal", ax = ax)
plt.title('Distribution of Race_5_merged_sum', fontsize = 18)
ax.set_xlabel("PotentialFraud")
ax.set_ylabel("Frequency")
ax.legend();
plt.show()

x=dev_copy[dev_copy['Race_5_merged_sum'] >= 20]

print("Value Counts of 1 and 0 :")
print(x['PotentialFraud'].value_counts())
print('Event Rate:%0.2f%%' %((x.PotentialFraud.value_counts()[1]/(x.PotentialFraud.value_counts()[0]+x.PotentialFraud.value_counts()[1]))*100))

print("So we created this flag which takes the value 1 if Race_5_merged_sum >=20 otherwise 0")


# In[ ]:





# In[128]:


derived_feat = [i for i in dev_copy.columns if i.startswith('derived') or i in (['Provider'])]


# In[129]:


derived_feat


# ## Feature creation using Weight of Evidence method

# In[71]:


import scorecardpy as sc


# In[72]:


# filter variable via missing rate, iv, identical value rate from raw development sample
dev_filter = sc.var_filter(dev, y="PotentialFraud")


# In[73]:


print("Removed Features: {}".format(list(set(dev.columns) - set(dev_filter.columns))))


# In[74]:


# woe binning ------
bins = sc.woebin(dev_filter, y="PotentialFraud")


# In[75]:


bins['Race_2_merged_sum']


# In[76]:


#bins[feat[10]]


# In[77]:


# For each raw variable in development sample, Convert each value with corresponding BIN woe
dev_woe = sc.woebin_ply(dev, bins)


# In[78]:


dev.head()


# In[79]:


dev_woe.head()


# In[80]:


woe_feat_dev = [i for i in dev_woe.columns if i.endswith('_woe') or i in (['Provider'])]


# In[81]:


#woe_feat_dev


# In[ ]:





# In[82]:


feat=dev.columns
feat=list(feat.drop(['Provider','PotentialFraud']))
#feat


# In[83]:


dev_iv=pd.DataFrame()
for i in range(0,len(feat)):
    dev_iv=dev_iv.append(pd.DataFrame(bins[feat[i]]))
dev_iv_final=dev_iv.groupby("variable")["total_iv"].agg(['mean'])
dev_iv_final.reset_index(level=0,inplace=True)
#dev_iv_final


# In[84]:


dev_iv_final.to_csv('C:\\Users\\Sahil\\Desktop\\Summer\\Project\\dev_iv_final.csv')


# # 4. Bringing everything together

# Merge all data and create the final feature list

# In[85]:


dev_final_1 =pd.merge(dev_copy[derived_feat], dev_woe[woe_feat_dev], on='Provider', how='left')           


# In[119]:


dev_final_1=pd.merge(dev_final_1,train_provider,  on='Provider', how='left')


# In[120]:


dev_final_1.head()


# In[124]:


dev_final_1.shape


# In[87]:


#check correlation


# **Now we will check for correlation between the counts and unique counts of features that are created by WOE Method** --
# because we know that counts and unique counts are highly correlated and if we take all of those variables for feature selection purpose or model development purpose then their collinearity will effect our further procedures and we will deviate from our result that we want. So we remove one between count and unique count from each of them and we keep that variable which has more correlation with PotentialFraud

# In[126]:


count_nunique_features=[i for i in dev_final_1.columns if i.startswith('ClmDiagnosisCode') or i.startswith('BeneID') or i.startswith('ClaimID')]


# In[127]:


len(count_nunique_features)


# In[121]:


corrMatrix = dev_final_1.corr()


# In[122]:


corrMatrix


# In[99]:


#corrMatrix[['PotentialFraud']][count_nunique_features]


# In[131]:


print(corrMatrix['PotentialFraud']['ClmDiagnosisCode_1_in_count_woe'])
print(corrMatrix['PotentialFraud']['ClmDiagnosisCode_1_in_nunique_woe'])


# In[ ]:


Here we take that variable which has the higher correlation with PotentialFraud


# In[117]:


corrMatrix['BeneID_count_woe']['BeneID_nunique_woe']


# In[118]:


corrMatrix['ClmDiagnosisCode_1_in_count_woe']['ClmDiagnosisCode_1_in_nunique_woe']


# In[92]:


selected_between_count_nunique_feat=pd.read_csv("C:\\Users\\Sahil\\Desktop\\Summer\\Project\\Selected from count and nunique.csv")
#selected_between_count_nunique_feat=list(selected_between_count_nunique_feat)
selected_between_count_nunique_feat


# In[93]:


selected_between_count_nunique_feat_list=list(selected_between_count_nunique_feat['selected variables'])
selected_between_count_nunique_feat_list


# In[94]:


to_delete_features=list(set(count_nunique_features)-set(selected_between_count_nunique_feat_list))
to_delete_features


# In[101]:


dev_final=dev_final_1.drop(columns=to_delete_features)


# In[102]:


dev_final.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[110]:


dev_final.shape


# In[109]:


dev_final=pd.merge(dev_final,train_provider, on='Provider', how='left')


# In[70]:


x_feat=dev_final.drop(["PotentialFraud",'Provider'], axis=1)


# In[71]:


x_feat.head()


# # Step 3: Feature Selection
# 
# ### This step will identify the important features that explains maximum variability of the model
# 
# - Select a small representive sample from the prepared data in last step
# - Get feature importance score/ set of top N features using different techniques like:
#   - WOE
#   - RFE
#   - Random Forest Feature importance score
#   - Catboost Feature importance score
#   - Decision Tree Feature importance score
# 
# <br>
# 
# *After getting their importances we save their cumulative score in a csv for each of the methods*
# 
# ### Output of this step is the list of top features to be used in the model

# #### Method-WOE

# In[72]:


# filter variable via missing rate, iv, identical value rate from raw development sample
dev_final_filter = sc.var_filter(dev_final, y="PotentialFraud")


# In[73]:


# woe binning ------
final_bins = sc.woebin(dev_final_filter, y="PotentialFraud")
#final_bins


# In[74]:


#final_bins['derived_ChronicCond_Cancer_flag']


# In[75]:


# For each raw variable in development sample, Convert each value with corresponding BIN woe
dev_final_woe = sc.woebin_ply(dev_final, final_bins)


# In[76]:


final_feat=dev_final_filter.columns
final_feat=list(final_feat.drop(['PotentialFraud']))
len(final_feat)


# In[77]:


dev_final.head()


# In[78]:


dev_final_iv=pd.DataFrame()
for i in range(0,len(final_feat)):
    dev_final_iv=dev_final_iv.append(pd.DataFrame(final_bins[final_feat[i]]))
dev_final_iv_final=dev_final_iv.groupby("variable")["total_iv"].agg(['mean'])
dev_final_iv_final.reset_index(level=0,inplace=True)
dev_final_iv_final


# In[79]:


dev_final_iv_final.to_csv('C:\\Users\\Sahil\\Desktop\\Summer\\Project\\dev_final_iv_final.csv')


# ### catboost method

# In[80]:


from catboost import CatBoostClassifier


# In[81]:


cat_model=CatBoostClassifier()
cat_model.fit(x_feat,dev_final['PotentialFraud'])


# In[82]:


cat_imp=pd.DataFrame()
cat_imp['Features']=list(x_feat.columns)
cat_imp['Importance']=list(cat_model.feature_importances_)
cat_imp=cat_imp.sort_values(by=['Importance'],ascending=False)
cat_imp['Cumulative']=cat_imp.Importance.cumsum()
cat_imp


# In[83]:


cat_imp.to_csv('C:\\Users\\Sahil\\Desktop\\Summer\\Project\\cat_imp_final.csv')


# ### random forest method

# In[84]:


from sklearn.ensemble import RandomForestClassifier


# In[85]:


rand_model=RandomForestClassifier()
rand_model.fit(x_feat,dev_final['PotentialFraud'])


# In[86]:


rand_imp=pd.DataFrame()
rand_imp['Features']=list(x_feat.columns)
rand_imp['Importance']=list(rand_model.feature_importances_)
rand_imp=rand_imp.sort_values(by=['Importance'],ascending=False)
rand_imp['Cumulative']=rand_imp.Importance.cumsum()
rand_imp


# In[87]:


rand_imp.to_csv('C:\\Users\\Sahil\\Desktop\\Summer\\Project\\rand_imp_final.csv')


# ### Decision Tree Method

# In[88]:


from sklearn.tree import DecisionTreeClassifier


# In[89]:


dt_model=DecisionTreeClassifier()
dt_model.fit(x_feat,dev_final['PotentialFraud'])


# In[90]:


dt_imp=pd.DataFrame()
dt_imp['Features']=list(x_feat.columns)
dt_imp['Importance']=list(dt_model.feature_importances_)
dt_imp=dt_imp.sort_values(by=['Importance'],ascending=False)
dt_imp['Cumulative']=dt_imp.Importance.cumsum()
dt_imp


# In[91]:


dt_imp.to_csv('C:\\Users\\Sahil\\Desktop\\Summer\\Project\\dt_imp_final.csv')


# ### RFE Model

# In[92]:


from sklearn.feature_selection import RFE


# In[93]:


rfe = RFE(estimator=rand_model,n_features_to_select=1)
rfe.fit(x_feat,dev_final['PotentialFraud'])


# In[94]:


rfe_imp=pd.DataFrame()
rfe_imp['features']=list(x_feat.columns)
rfe_imp['rank']=list(rfe.ranking_)
rfe_imp=rfe_imp.sort_values(by=['rank'],ascending=True)
rfe_imp


# In[95]:


rfe_imp.to_csv('C:\\Users\\Sahil\\Desktop\\Summer\\Project\\rfe_imp_final.csv')


# ## Selection of Top Features
# Now as we have got the feature importances from each of the Methods stated previously, Now we select features by comparing them. First RFE method gives us ranks. Next we observe those features whose cumulative impoertance is below 80% for  Random Forest, Decision Tree and catboost method & whose Information value is greater than 0.2. Then considering ranks as pivots we select those features  for which at least 3 out of 4 conditions are satisfied.
# 
# *we have done this in an excel file manually and saved it into  csv file named 'selected_features'.*

# # Step 4: Model Training and Hyperparameter Tuning
# #### In this step we will prepare the final model using top features selected from previous step
# 
# - Train different ML Algorithms with default parameters and compare performance KPIs - AUC, Accuracy, Precision, Recall
# - **Bias-Variance Trade-off (Compare Train-Test KPIs)**:
#     For building a good model we want both bias and variance to be low which is not possible to attain. So we need to find a  proper balance between these two. Here we take a range of values of a particular parameter for a model and get an insight       about the range where the bias and variance of the model are optimized.
# 
# - Select best model and perform hyper-parameter tuning
# - Select optimal set of parameters and finalize model
# 
# ### Output of this step is final model pkl file

# In[106]:


# first we read the csv file of the selected features
selected_features=pd.read_csv("C:\\Users\\Sahil\\Desktop\\Summer\\Project\\selected features.csv")


# In[107]:


final_features=list(selected_features['Selected Features'])
#final_features


# In[111]:


final_train=dev_final[final_features]


# In[112]:


final_train.shape


# In[113]:


top_features=list(set(final_train.columns)-set(['Provider'])-set(['PotentialFraud']))
#top_features


# In[114]:


final_train.to_csv("C:\\Users\\Sahil\\Desktop\\Summer\\Project\\final_train.csv")


# In[115]:


final_train.head()


# In[116]:


final_train.shape


# In[132]:


#Dividing the model into the train_test split

x_train,x_test,y_train,y_test=train_test_split(final_train[top_features], final_train['PotentialFraud'], 
                                                    test_size = 0.2,random_state = 0, stratify = final_train['PotentialFraud'])


# In[133]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# ## Logistic Regression

# In[134]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression


# In[135]:


lr = LogisticRegression(class_weight='balanced')

lr.fit(x_train, y_train)

y_test_pred = lr.predict(x_test)
y_train_pred = lr.predict(x_train)

print("========================== Logistic Regression - Model Report =========================")

print("===================== Model Accuracy ==============================")
print("Train Accuracy: %0.3f"%(lr.score(x_train, y_train)))
print("Test Accuracy: %0.3f"%(lr.score(x_test, y_test)))
print("=====================================================")
# Cross validation
from sklearn.model_selection import cross_val_score

print("===================== Cross Validation Scores ===========================")
cv_results_lr = cross_val_score(lr, x_train, y_train, cv=5, scoring='roc_auc')
for i in range(len(cv_results_lr)):
    print("Fold - {}".format(i+1))
    print("CV Score = %0.3f"%cv_results_lr[i])
print("Average 5-Fold CV Score: %0.3f"%(np.mean(cv_results_lr)))
print("=====================================================")

# Performance Metrices
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

print("===================== Model Performance Metrices ===========================")
print("Test Confusion Matrix: ")
print(confusion_matrix(y_test, y_test_pred))
print("============================")
print("Train Confusion Matrix: ")
print(confusion_matrix(y_train, y_train_pred))

print("=====================================================")
print("Test Classification Report: ")
print(classification_report(y_test, y_test_pred))
print("=====================================================")
print("Train Classification Report: ")
print(classification_report(y_train, y_train_pred))

y_test_pred_prob = lr.predict_proba(x_test)[:,1]
y_train_pred_prob = lr.predict_proba(x_train)[:,1]

print("===================== Model AUC Scores ===========================")
print("Train AUC Score: %0.3f"%(roc_auc_score(y_train, y_train_pred_prob)))
print("Test AUC Score: %0.3f"%(roc_auc_score(y_test, y_test_pred_prob)))
print("=====================================================")

print("===================== Model AUC Curve ===========================")
fpr, tpr, threshold = roc_curve(y_test, y_test_pred_prob)
fpr1, tpr1, threshold1 = roc_curve(y_train, y_train_pred_prob)

plt.figure(figsize=(10,5))
plt.title('Logistic Regression ROC Curve')
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr1, tpr1, label='Logistic Regression - Train')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression - Test')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
plt.show()


# ## Decision Tree

# In[136]:


from sklearn.tree import DecisionTreeClassifier


# In[137]:


dt = DecisionTreeClassifier(class_weight='balanced')

dt.fit(x_train, y_train)

y_test_pred = dt.predict(x_test)
y_train_pred = dt.predict(x_train)

print("========================== Decision Tree - Model Report =========================")

print("===================== Model Accuracy ==============================")
print("Train Accuracy: %0.3f"%(dt.score(x_train, y_train)))
print("Test Accuracy: %0.3f"%(dt.score(x_test, y_test)))
print("=====================================================")
# Cross validation
from sklearn.model_selection import cross_val_score

print("===================== Cross Validation Scores ===========================")
cv_results_dt = cross_val_score(dt, x_train, y_train, cv=5, scoring='roc_auc')
for i in range(len(cv_results_dt)):
    print("Fold - {}".format(i+1))
    print("CV Score = %0.3f"%cv_results_dt[i])
print("Average 5-Fold CV Score: %0.3f"%(np.mean(cv_results_dt)))
print("=====================================================")

# Performance Metrices
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

print("===================== Model Performance Metrices ===========================")
print("Test Confusion Matrix: ")
print(confusion_matrix(y_test, y_test_pred))
print("============================")
print("Train Confusion Matrix: ")
print(confusion_matrix(y_train, y_train_pred))

print("=====================================================")
print("Test Classification Report: ")
print(classification_report(y_test, y_test_pred))
print("=====================================================")
print("Train Classification Report: ")
print(classification_report(y_train, y_train_pred))

y_test_pred_prob = dt.predict_proba(x_test)[:,1]
y_train_pred_prob = dt.predict_proba(x_train)[:,1]

print("===================== Model AUC Scores ===========================")
print("Train AUC Score: %0.3f"%(roc_auc_score(y_train, y_train_pred_prob)))
print("Test AUC Score: %0.3f"%(roc_auc_score(y_test, y_test_pred_prob)))
print("=====================================================")

print("===================== Model AUC Curve ===========================")
fpr, tpr, threshold = roc_curve(y_test, y_test_pred_prob)
fpr1, tpr1, threshold1 = roc_curve(y_train, y_train_pred_prob)

plt.figure(figsize=(10,5))
plt.title('Decision Tree ROC Curve')
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr1, tpr1, label='Decision Tree - Train')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Decision Tree - Test')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
plt.show()


# #### Bias-Variance Trade off

# In[138]:


train_score = []
test_score = []
vals = []

# To check the bias-variance take any parameters and iterate over a few values - Like n_estimator, Max_depth
for k in np.arange(2,40,2):
    vals.append(k)
    dt= DecisionTreeClassifier(class_weight='balanced', min_samples_leaf=k)
    dt.fit(x_train, y_train)
    
    tr_score = roc_auc_score(y_train, dt.predict_proba(x_train)[:,1])
    train_score.append(tr_score)
    
    te_score = roc_auc_score(y_test, dt.predict_proba(x_test)[:,1])
    test_score.append(te_score)


# In[139]:


plt.figure(figsize=(10,5))
plt.xlabel('Different Values of max_depth')
plt.ylabel('Model score')
plt.plot(vals, train_score, color = 'r', label = "training score")
plt.plot(vals, test_score, color = 'b', label = 'test score')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()


# ### Hyperparameter tuning

# In[140]:


from sklearn.model_selection import GridSearchCV


# In[141]:


dt.get_params()


# In[142]:


params_dt = {'max_depth': [2,3,4,5,6],
             'min_samples_split':[10,20,25,30],
             'min_samples_leaf':[12,14,17,25]
              }


# In[143]:


grid_dt = GridSearchCV(estimator  = dt,
                         param_grid = params_dt,
                         scoring    = 'roc_auc',
                         cv         = 5,
                         n_jobs     = -1,
                         verbose    = True
                        )


# In[144]:


grid_dt.fit(x_train, y_train)


# In[145]:


print("Best Hyperparameters: \n", grid_dt.best_params_)
print("Best AUC Score: \n", grid_dt.best_score_)


# In[146]:


# Save the best combination of parameters as best model
best_model = grid_dt.best_estimator_


# In[147]:


y_test_pred = best_model.predict(x_test)
y_train_pred = best_model.predict(x_train)


print("========================== Decision Tree - Model Report =========================")

print("===================== Model Accuracy ==============================")
print("Train Accuracy: %0.3f"%(best_model.score(x_train, y_train)))
print("Test Accuracy: %0.3f"%(best_model.score(x_test, y_test)))
print("=====================================================")

# Performance Metrices
# from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

print("===================== Model Performance Metrices ===========================")
print("Test Confusion Matrix: ")
print(confusion_matrix(y_test, y_test_pred))
print("============================")
print("Train Confusion Matrix: ")
print(confusion_matrix(y_train, y_train_pred))

print("=====================================================")
print("Test Classification Report: ")
print(classification_report(y_test, y_test_pred))
print("=====================================================")
print("Train Classification Report: ")
print(classification_report(y_train, y_train_pred))

y_test_pred_prob = best_model.predict_proba(x_test)[:,1]
y_train_pred_prob = best_model.predict_proba(x_train)[:,1]

print("===================== Model AUC Scores ===========================")
print("Train AUC Score: %0.3f"%(roc_auc_score(y_train, y_train_pred_prob)))
print("Test AUC Score: %0.3f"%(roc_auc_score(y_test, y_test_pred_prob)))
print("=====================================================")

print("===================== Model AUC Curve ===========================")
fpr, tpr, threshold = roc_curve(y_test, y_test_pred_prob)
fpr1, tpr1, threshold1 = roc_curve(y_train, y_train_pred_prob)

plt.figure(figsize=(10,5))
plt.title('Decision Tree ROC Curve')
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr1, tpr1, label='Decision Tree Tuned - Train')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Decision Tree Tuned - Test')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# ## Random Forest

# In[148]:


from sklearn.ensemble import RandomForestClassifier


# In[149]:


rf = RandomForestClassifier(class_weight='balanced')

rf.fit(x_train, y_train)

y_test_pred = rf.predict(x_test)
y_train_pred = rf.predict(x_train)

print("========================== Random Forest - Model Report =========================")

print("===================== Model Accuracy ==============================")
print("Train Accuracy: %0.3f"%(rf.score(x_train, y_train)))
print("Test Accuracy: %0.3f"%(rf.score(x_test, y_test)))
print("=====================================================")

# Cross validation
# from sklearn.model_selection import cross_val_score

print("===================== Cross Validation Scores ===========================")
cv_results_rf = cross_val_score(rf, x_train, y_train, cv=5, scoring='roc_auc')
for i in range(len(cv_results_rf)):
    print("Fold - {}".format(i+1))
    print("CV Score = %0.3f"%cv_results_rf[i])
print("Average 5-Fold CV Score: %0.3f"%(np.mean(cv_results_rf)))
print("=====================================================")

# Performance Metrices
# from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

print("===================== Model Performance Metrices ===========================")
print("Test Confusion Matrix: ")
print(confusion_matrix(y_test, y_test_pred))
print("============================")
print("Train Confusion Matrix: ")
print(confusion_matrix(y_train, y_train_pred))

print("=====================================================")
print("Test Classification Report: ")
print(classification_report(y_test, y_test_pred))
print("=====================================================")
print("Train Classification Report: ")
print(classification_report(y_train, y_train_pred))

y_test_pred_prob = rf.predict_proba(x_test)[:,1]
y_train_pred_prob = rf.predict_proba(x_train)[:,1]

print("===================== Model AUC Scores ===========================")
print("Train AUC Score: %0.3f"%(roc_auc_score(y_train, y_train_pred_prob)))
print("Test AUC Score: %0.3f"%(roc_auc_score(y_test, y_test_pred_prob)))
print("=====================================================")

print("===================== Model AUC Curve ===========================")
fpr, tpr, threshold = roc_curve(y_test, y_test_pred_prob)
fpr1, tpr1, threshold1 = roc_curve(y_train, y_train_pred_prob)

plt.figure(figsize=(10,5))
plt.title('Random Forest ROC Curve')
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr1, tpr1, label='Random Forest - Train')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Random Forest - Test')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
plt.show()


# #### Bias_variance tradeoff

# In[199]:


train_score = []
test_score = []
vals = []

for k in np.arange(1,15,1):
    vals.append(k)
    rf= RandomForestClassifier(class_weight='balanced',max_depth=k)
    rf.fit(x_train, y_train)
    
    tr_score = roc_auc_score(y_train, rf.predict_proba(x_train)[:,1])
    train_score.append(tr_score)
    
    te_score = roc_auc_score(y_test, rf.predict_proba(x_test)[:,1])
    test_score.append(te_score)
    
    
plt.figure(figsize=(10,5))
plt.xlabel('Different Values of max_depth')
plt.ylabel('Model score')
plt.plot(vals, train_score, color = 'r', label = "training score")
plt.plot(vals, test_score, color = 'b', label = 'test score')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()


# The graph above shows model score against the parameter max_depth varying from 1 to 15.
# This graph shows after value 3 the train AUC increases but test AUC decreases. So during hyperparameter tuning we varied max_depth from 2 to 5.

# In[200]:


train_score = []
test_score = []
vals = []

# To check the bias-variance take any parameters and iterate over a few values - Like n_estimator, Max_depth
for k in np.arange(50,300,25):
    vals.append(k)
    rf= RandomForestClassifier(class_weight='balanced',n_estimators=k,max_depth=3)
    rf.fit(x_train, y_train)
    
    tr_score = roc_auc_score(y_train, rf.predict_proba(x_train)[:,1])
    train_score.append(tr_score)
    
    te_score = roc_auc_score(y_test, rf.predict_proba(x_test)[:,1])
    test_score.append(te_score)
    
plt.figure(figsize=(10,5))
plt.xlabel('Different Values of n_estimators')
plt.ylabel('Model score')
plt.plot(vals, train_score, color = 'r', label = "training score")
plt.plot(vals, test_score, color = 'b', label = 'test score')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()


# The above graph shows model score against the parameter n_estimators varying from 50 to 300.
# This graph shows where train AUC increases but test AUC decreases. So during hyperparameter tuning we varied max_depth from 50 to 225.

# In[202]:


train_score = []
test_score = []
vals = []

# To check the bias-variance take any parameters and iterate over a few values - Like n_estimator, Max_depth
for k in np.arange(2,15,1):
    vals.append(k)
    rf= RandomForestClassifier(class_weight='balanced',min_samples_split=k,max_depth=3)
    rf.fit(x_train, y_train)
    
    tr_score = roc_auc_score(y_train, rf.predict_proba(x_train)[:,1])
    train_score.append(tr_score)
    
    te_score = roc_auc_score(y_test, rf.predict_proba(x_test)[:,1])
    test_score.append(te_score)
    
plt.figure(figsize=(10,5))
plt.xlabel('Different Values of min_samples_split')
plt.ylabel('Model score')
plt.plot(vals, train_score, color = 'r', label = "training score")
plt.plot(vals, test_score, color = 'b', label = 'test score')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()


# In[203]:


train_score = []
test_score = []
vals = []

# To check the bias-variance take any parameters and iterate over a few values - Like n_estimator, Max_depth
for k in np.arange(1,10,1):
    vals.append(k)
    rf= RandomForestClassifier(class_weight='balanced',min_samples_leaf=k)
    rf.fit(x_train, y_train)
    
    tr_score = roc_auc_score(y_train, rf.predict_proba(x_train)[:,1])
    train_score.append(tr_score)
    
    te_score = roc_auc_score(y_test, rf.predict_proba(x_test)[:,1])
    test_score.append(te_score)
    
plt.figure(figsize=(10,5))
plt.xlabel('Different Values of min_samples_leaf')
plt.ylabel('Model score')
plt.plot(vals, train_score, color = 'r', label = "training score")
plt.plot(vals, test_score, color = 'b', label = 'test score')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()


# In[ ]:


train_score = []
test_score = []
vals = []

# To check the bias-variance take any parameters and iterate over a few values - Like n_estimator, Max_depth
for k in np.arange(1,10,1):
    vals.append(k)
    rf= RandomForestClassifier(class_weight='balanced',min_samples_leaf=k)
    rf.fit(x_train, y_train)
    
    tr_score = roc_auc_score(y_train, rf.predict_proba(x_train)[:,1])
    train_score.append(tr_score)
    
    te_score = roc_auc_score(y_test, rf.predict_proba(x_test)[:,1])
    test_score.append(te_score)
    
plt.figure(figsize=(10,5))
plt.xlabel('Different Values of min_samples_leaf')
plt.ylabel('Model score')
plt.plot(vals, train_score, color = 'r', label = "training score")
plt.plot(vals, test_score, color = 'b', label = 'test score')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()


# ### Hyperparameter Tuning

# In[152]:


rf.get_params()


# In[204]:


params_rf = {'max_depth': [2,3,4,5,6],
               'min_samples_split':[2,3,4,6],
               'min_samples_leaf':[1,2,3],
               'n_estimators':[50,75,100,130,225],
              }


# In[205]:


grid_rf = GridSearchCV(estimator  = rf,
                         param_grid = params_rf,
                         scoring    = 'roc_auc',
                         cv         = 5,
                         n_jobs     = -1,
                         verbose    = True
                        )


# In[206]:


grid_rf.fit(x_train, y_train)


# In[207]:


print("Best Hyperparameters: \n", grid_rf.best_params_)
print("Best AUC Score: \n", grid_rf.best_score_)


# In[208]:


# Save the best combination of parameters as best model
best_model_rf = grid_rf.best_estimator_


# In[209]:


y_test_pred = best_model_rf.predict(x_test)
y_train_pred = best_model_rf.predict(x_train)


print("========================== Random Forest Tuned - Model Report =========================")

print("===================== Model Accuracy ==============================")
print("Train Accuracy: %0.3f"%(best_model_rf.score(x_train, y_train)))
print("Test Accuracy: %0.3f"%(best_model_rf.score(x_test, y_test)))
print("=====================================================")

# Performance Metrices
# from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

print("===================== Model Performance Metrices ===========================")
print("Test Confusion Matrix: ")
print(confusion_matrix(y_test, y_test_pred))
print("============================")
print("Train Confusion Matrix: ")
print(confusion_matrix(y_train, y_train_pred))

print("=====================================================")
print("Test Classification Report: ")
print(classification_report(y_test, y_test_pred))
print("=====================================================")
print("Train Classification Report: ")
print(classification_report(y_train, y_train_pred))

y_test_pred_prob = best_model_rf.predict_proba(x_test)[:,1]
y_train_pred_prob = best_model_rf.predict_proba(x_train)[:,1]

print("===================== Model AUC Scores ===========================")
print("Train AUC Score: %0.3f"%(roc_auc_score(y_train, y_train_pred_prob)))
print("Test AUC Score: %0.3f"%(roc_auc_score(y_test, y_test_pred_prob)))
print("=====================================================")

print("===================== Model AUC Curve ===========================")
fpr, tpr, threshold = roc_curve(y_test, y_test_pred_prob)
fpr1, tpr1, threshold1 = roc_curve(y_train, y_train_pred_prob)

plt.figure(figsize=(10,5))
plt.title('Random Forest Tuned ROC Curve')
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr1, tpr1, label='Random Forest Tuned - Train')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Random Forest Tuned - Test')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# ###  setting the max depth =3

# In[159]:


rf = RandomForestClassifier(class_weight='balanced', max_depth=3)

rf.fit(x_train, y_train)

y_test_pred = rf.predict(x_test)
y_train_pred = rf.predict(x_train)

print("========================== Random Forest - Model Report =========================")

print("===================== Model Accuracy ==============================")
print("Train Accuracy: %0.3f"%(rf.score(x_train, y_train)))
print("Test Accuracy: %0.3f"%(rf.score(x_test, y_test)))
print("=====================================================")

# Cross validation
# from sklearn.model_selection import cross_val_score

print("===================== Cross Validation Scores ===========================")
cv_results_rf = cross_val_score(rf, x_train, y_train, cv=5, scoring='roc_auc')
for i in range(len(cv_results_rf)):
    print("Fold - {}".format(i+1))
    print("CV Score = %0.3f"%cv_results_rf[i])
print("Average 5-Fold CV Score: %0.3f"%(np.mean(cv_results_rf)))
print("=====================================================")

# Performance Metrices
# from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

print("===================== Model Performance Metrices ===========================")
print("Test Confusion Matrix: ")
print(confusion_matrix(y_test, y_test_pred))
print("============================")
print("Train Confusion Matrix: ")
print(confusion_matrix(y_train, y_train_pred))

print("=====================================================")
print("Test Classification Report: ")
print(classification_report(y_test, y_test_pred))
print("=====================================================")
print("Train Classification Report: ")
print(classification_report(y_train, y_train_pred))

y_test_pred_prob = rf.predict_proba(x_test)[:,1]
y_train_pred_prob = rf.predict_proba(x_train)[:,1]

print("===================== Model AUC Scores ===========================")
print("Train AUC Score: %0.3f"%(roc_auc_score(y_train, y_train_pred_prob)))
print("Test AUC Score: %0.3f"%(roc_auc_score(y_test, y_test_pred_prob)))
print("=====================================================")

print("===================== Model AUC Curve ===========================")
fpr, tpr, threshold = roc_curve(y_test, y_test_pred_prob)
fpr1, tpr1, threshold1 = roc_curve(y_train, y_train_pred_prob)

plt.figure(figsize=(10,5))
plt.title('Random Forest ROC Curve')
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr1, tpr1, label='Random Forest - Train')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Random Forest - Test')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
plt.show()


# # Light GBM

# In[160]:


from lightgbm import LGBMClassifier


# In[161]:


lgbm = LGBMClassifier(class_weight='balanced')

lgbm.fit(x_train, y_train)

y_test_pred = lgbm.predict(x_test)
y_train_pred = lgbm.predict(x_train)

print("========================== Light GBM - Model Report =========================")

print("===================== Model Accuracy ==============================")
print("Train Accuracy: %0.3f"%(lgbm.score(x_train, y_train)))
print("Test Accuracy: %0.3f"%(lgbm.score(x_test, y_test)))
print("=====================================================")

# Cross validation
# from sklearn.model_selection import cross_val_score

print("===================== Cross Validation Scores ===========================")
cv_results_lgbm = cross_val_score(lgbm, x_train, y_train, cv=5, scoring='roc_auc')
for i in range(len(cv_results_lgbm)):
    print("Fold - {}".format(i+1))
    print("CV Score = %0.3f"%cv_results_lgbm[i])
print("Average 5-Fold CV Score: %0.3f"%(np.mean(cv_results_lgbm)))
print("=====================================================")

# Performance Metrices
# from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

print("===================== Model Performance Metrices ===========================")
print("Test Confusion Matrix: ")
print(confusion_matrix(y_test, y_test_pred))
print("============================")
print("Train Confusion Matrix: ")
print(confusion_matrix(y_train, y_train_pred))

print("=====================================================")
print("Test Classification Report: ")
print(classification_report(y_test, y_test_pred))
print("=====================================================")
print("Train Classification Report: ")
print(classification_report(y_train, y_train_pred))

y_test_pred_prob = lgbm.predict_proba(x_test)[:,1]
y_train_pred_prob = lgbm.predict_proba(x_train)[:,1]

print("===================== Model AUC Scores ===========================")
print("Train AUC Score: %0.3f"%(roc_auc_score(y_train, y_train_pred_prob)))
print("Test AUC Score: %0.3f"%(roc_auc_score(y_test, y_test_pred_prob)))
print("=====================================================")

print("===================== Model AUC Curve ===========================")
fpr, tpr, threshold = roc_curve(y_test, y_test_pred_prob)
fpr1, tpr1, threshold1 = roc_curve(y_train, y_train_pred_prob)

plt.figure(figsize=(10,5))
plt.title('Light GBM ROC Curve')
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr1, tpr1, label='Light GBM - Train')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Light GBM - Test')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
plt.show()


# ### Bias-Variance trade-off

# In[192]:


# varying max_depth
train_score = []
test_score = []
vals = []


for k in np.arange(1,15,1):
    vals.append(k)
    lgbm= LGBMClassifier(class_weight='balanced',max_depth =k)
    lgbm.fit(x_train, y_train)
    
    tr_score = roc_auc_score(y_train, lgbm.predict_proba(x_train)[:,1])
    train_score.append(tr_score)
    
    te_score = roc_auc_score(y_test, lgbm.predict_proba(x_test)[:,1])
    test_score.append(te_score)
    
    
plt.figure(figsize=(10,5))
plt.xlabel('Different Values of max_depth')
plt.ylabel('Model score')
plt.plot(vals, train_score, color = 'r', label = "training score")
plt.plot(vals, test_score, color = 'b', label = 'test score')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()


# In[193]:


# varying n_estimators
train_score = []
test_score = []
vals = []


for k in np.arange(5,200,10):
    vals.append(k)
    lgbm= LGBMClassifier(class_weight='balanced',n_estimators =k)
    lgbm.fit(x_train, y_train)
    
    tr_score = roc_auc_score(y_train, lgbm.predict_proba(x_train)[:,1])
    train_score.append(tr_score)
    
    te_score = roc_auc_score(y_test, lgbm.predict_proba(x_test)[:,1])
    test_score.append(te_score)
    
    
plt.figure(figsize=(10,5))
plt.xlabel('Different Values of n_estimators')
plt.ylabel('Model score')
plt.plot(vals, train_score, color = 'r', label = "training score")
plt.plot(vals, test_score, color = 'b', label = 'test score')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()


# In[197]:


# varying learning_rate
train_score = []
test_score = []
vals = []


for k in np.arange(0.01,0.3,0.05):
    vals.append(k)
    lgbm= LGBMClassifier(class_weight='balanced',learning_rate =k)
    lgbm.fit(x_train, y_train)
    
    tr_score = roc_auc_score(y_train, lgbm.predict_proba(x_train)[:,1])
    train_score.append(tr_score)
    
    te_score = roc_auc_score(y_test, lgbm.predict_proba(x_test)[:,1])
    test_score.append(te_score)
    
    
plt.figure(figsize=(10,5))
plt.xlabel('Different Values of learning_rate')
plt.ylabel('Model score')
plt.plot(vals, train_score, color = 'r', label = "training score")
plt.plot(vals, test_score, color = 'b', label = 'test score')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()


# In[198]:


# varying min_child_samples
train_score = []
test_score = []
vals = []


for k in np.arange(1,15,1):
    vals.append(k)
    lgbm= LGBMClassifier(class_weight='balanced',min_child_samples =k)
    lgbm.fit(x_train, y_train)
    
    tr_score = roc_auc_score(y_train, lgbm.predict_proba(x_train)[:,1])
    train_score.append(tr_score)
    
    te_score = roc_auc_score(y_test, lgbm.predict_proba(x_test)[:,1])
    test_score.append(te_score)
    
    
plt.figure(figsize=(10,5))
plt.xlabel('Different Values of min_child_samples')
plt.ylabel('Model score')
plt.plot(vals, train_score, color = 'r', label = "training score")
plt.plot(vals, test_score, color = 'b', label = 'test score')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()


# In[162]:


train_score = []
test_score = []
vals = []

# To check the bias-variance take any parameters and iterate over a few values - Like n_estimator, Max_depth
for k in np.arange(5,200,15):
    vals.append(k)
    lgbm= LGBMClassifier(class_weight='balanced', n_estimators = k,min_child_samples=5,max_depth=2)
    lgbm.fit(x_train, y_train)
    
    tr_score = roc_auc_score(y_train, lgbm.predict_proba(x_train)[:,1])
    train_score.append(tr_score)
    
    te_score = roc_auc_score(y_test, lgbm.predict_proba(x_test)[:,1])
    test_score.append(te_score)


# In[163]:


plt.figure(figsize=(10,5))
plt.xlabel('Different Values of max_depth')
plt.ylabel('Model score')
plt.plot(vals, train_score, color = 'r', label = "training score")
plt.plot(vals, test_score, color = 'b', label = 'test score')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()


# ### hyperparameter tuning

# In[164]:


from sklearn.model_selection import GridSearchCV


# In[165]:


lgbm.get_params()


# In[166]:


params_lgbm = {'max_depth': [1,2,3,4,5],
               'learning_rate':[0.05,0.2,0.5,0.7],
               'min_child_samples':[5,15,30, 40,50],
               'n_estimators':[20,30,40,50],
              }


# In[167]:


grid_lgbm = GridSearchCV(estimator  = lgbm,
                         param_grid = params_lgbm,
                         scoring    = 'roc_auc',
                         cv         = 3,
                         n_jobs     = -1,
                         verbose    = True
                        )


# In[168]:


grid_lgbm.fit(x_train, y_train)


# In[169]:


print("Best Hyperparameters: \n", grid_lgbm.best_params_)
print("Best AUC Score: \n", grid_lgbm.best_score_)


# In[170]:


# Save the best combination of parameters as best model
best_model_lgbm = grid_lgbm.best_estimator_


# In[171]:


y_test_pred = best_model_lgbm.predict(x_test)
y_train_pred = best_model_lgbm.predict(x_train)


print("========================== Light GBM Tuned - Model Report =========================")

print("===================== Model Accuracy ==============================")
print("Train Accuracy: %0.3f"%(best_model_lgbm.score(x_train, y_train)))
print("Test Accuracy: %0.3f"%(best_model_lgbm.score(x_test, y_test)))
print("=====================================================")

# Performance Metrices
# from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

print("===================== Model Performance Metrices ===========================")
print("Test Confusion Matrix: ")
print(confusion_matrix(y_test, y_test_pred))
print("============================")
print("Train Confusion Matrix: ")
print(confusion_matrix(y_train, y_train_pred))

print("=====================================================")
print("Test Classification Report: ")
print(classification_report(y_test, y_test_pred))
print("=====================================================")
print("Train Classification Report: ")
print(classification_report(y_train, y_train_pred))

y_test_pred_prob = best_model_lgbm.predict_proba(x_test)[:,1]
y_train_pred_prob = best_model_lgbm.predict_proba(x_train)[:,1]

print("===================== Model AUC Scores ===========================")
print("Train AUC Score: %0.3f"%(roc_auc_score(y_train, y_train_pred_prob)))
print("Test AUC Score: %0.3f"%(roc_auc_score(y_test, y_test_pred_prob)))
print("=====================================================")

print("===================== Model AUC Curve ===========================")
fpr, tpr, threshold = roc_curve(y_test, y_test_pred_prob)
fpr1, tpr1, threshold1 = roc_curve(y_train, y_train_pred_prob)

plt.figure(figsize=(10,5))
plt.title('Light GBM Tuned ROC Curve')
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr1, tpr1, label='Light GBM Tuned - Train')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Light GBM Tuned - Test')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
plt.show()


# # XGBoost

# In[172]:


from xgboost import XGBClassifier


# In[173]:


xgb = XGBClassifier()

xgb.fit(x_train, y_train)

y_test_pred = xgb.predict(x_test)
y_train_pred = xgb.predict(x_train)

print("========================== XG Boost - Model Report =========================")

print("===================== Model Accuracy ==============================")
print("Train Accuracy: %0.3f"%(xgb.score(x_train, y_train)))
print("Test Accuracy: %0.3f"%(xgb.score(x_test, y_test)))
print("=====================================================")

# Cross validation
# from sklearn.model_selection import cross_val_score

print("===================== Cross Validation Scores ===========================")
cv_results_xgb = cross_val_score(xgb, x_train, y_train, cv=5, scoring='roc_auc')
for i in range(len(cv_results_xgb)):
    print("Fold - {}".format(i+1))
    print("CV Score = %0.3f"%cv_results_xgb[i])
print("Average 5-Fold CV Score: %0.3f"%(np.mean(cv_results_xgb)))
print("=====================================================")

# Performance Metrices
# from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

print("===================== Model Performance Metrices ===========================")
print("Test Confusion Matrix: ")
print(confusion_matrix(y_test, y_test_pred))
print("============================")
print("Train Confusion Matrix: ")
print(confusion_matrix(y_train, y_train_pred))

print("=====================================================")
print("Test Classification Report: ")
print(classification_report(y_test, y_test_pred))
print("=====================================================")
print("Train Classification Report: ")
print(classification_report(y_train, y_train_pred))

y_test_pred_prob = xgb.predict_proba(x_test)[:,1]
y_train_pred_prob = xgb.predict_proba(x_train)[:,1]

print("===================== Model AUC Scores ===========================")
print("Train AUC Score: %0.3f"%(roc_auc_score(y_train, y_train_pred_prob)))
print("Test AUC Score: %0.3f"%(roc_auc_score(y_test, y_test_pred_prob)))
print("=====================================================")

print("===================== Model AUC Curve ===========================")
fpr, tpr, threshold = roc_curve(y_test, y_test_pred_prob)
fpr1, tpr1, threshold1 = roc_curve(y_train, y_train_pred_prob)

plt.figure(figsize=(10,5))
plt.title('XG Boost ROC Curve')
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr1, tpr1, label='XG Boost - Train')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='XG Boost - Test')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
plt.show()


# This model is overfitted so we have to go for hyperparameter tuning. Before that we need to check bias variance tradeoff

# ### bias variance trade off

# In[187]:


# varying max_depth
train_score = []
test_score = []
vals = []

# To check the bias-variance take any parameters and iterate over a few values - Like n_estimator, Max_depth
for k in np.arange(1,15,1):
    vals.append(k)
    xgb= XGBClassifier(max_depth =k)
    xgb.fit(x_train, y_train)
    
    tr_score = roc_auc_score(y_train, xgb.predict_proba(x_train)[:,1])
    train_score.append(tr_score)
    
    te_score = roc_auc_score(y_test, xgb.predict_proba(x_test)[:,1])
    test_score.append(te_score)
    
    
plt.figure(figsize=(10,5))
plt.xlabel('Different Values of max_depth')
plt.ylabel('Model score')
plt.plot(vals, train_score, color = 'r', label = "training score")
plt.plot(vals, test_score, color = 'b', label = 'test score')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()


# The above graph tells us that after 4 train_AUC increases but test_AUC decreases. So we must check max_depth from 2 to 4 where the model performs best

# In[188]:


train_score = []
test_score = []
vals = []

# To check the bias-variance take any parameters and iterate over a few values - Like n_estimator, Max_depth
for k in np.arange(0.1,0.9,0.1):
    vals.append(k)
    xgb= XGBClassifier(subsample=k,max_depth = 3)
    xgb.fit(x_train, y_train)
    
    tr_score = roc_auc_score(y_train, xgb.predict_proba(x_train)[:,1])
    train_score.append(tr_score)
    
    te_score = roc_auc_score(y_test, xgb.predict_proba(x_test)[:,1])
    test_score.append(te_score)
    
    
plt.figure(figsize=(10,5))
plt.xlabel('Different Values of subsample for fixed max_depth=3')
plt.ylabel('Model score')
plt.plot(vals, train_score, color = 'r', label = "training score")
plt.plot(vals, test_score, color = 'b', label = 'test score')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()


# At subsample=0.6 both AUC values are high so we need check in neighbourhood region

# In[189]:


train_score = []
test_score = []
vals = []

# To check the bias-variance take any parameters and iterate over a few values - Like n_estimator, Max_depth
for k in np.arange(10,200,15):
    vals.append(k)
    xgb= XGBClassifier(n_estimators=k,max_depth = 3)
    xgb.fit(x_train, y_train)
    
    tr_score = roc_auc_score(y_train, xgb.predict_proba(x_train)[:,1])
    train_score.append(tr_score)
    
    te_score = roc_auc_score(y_test, xgb.predict_proba(x_test)[:,1])
    test_score.append(te_score)
    
    
plt.figure(figsize=(10,5))
plt.xlabel('Different Values of n_estimators for fixed max_depth=3')
plt.ylabel('Model score')
plt.plot(vals, train_score, color = 'r', label = "training score")
plt.plot(vals, test_score, color = 'b', label = 'test score')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()


# In[191]:


train_score = []
test_score = []
vals = []


for k in np.arange(1,15,1):
    vals.append(k)
    xgb= XGBClassifier(min_child_weight=k,max_depth = 3, n_estimators=100)
    xgb.fit(x_train, y_train)
    
    tr_score = roc_auc_score(y_train, xgb.predict_proba(x_train)[:,1])
    train_score.append(tr_score)
    
    te_score = roc_auc_score(y_test, xgb.predict_proba(x_test)[:,1])
    test_score.append(te_score)
    
    
plt.figure(figsize=(10,5))
plt.xlabel('Different Values of min_child_weight for fixed max_depth=3')
plt.ylabel('Model score')
plt.plot(vals, train_score, color = 'r', label = "training score")
plt.plot(vals, test_score, color = 'b', label = 'test score')
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()


# **From the other parameters we got some feasible range of values for hyperparameters. Now we are to perform grid search to find the best possible combination of hyperparameters.**

# ### Hyper parameter Tuning

# In[176]:


xgb.get_params()


# In[177]:


params_xgb = {'max_depth': [1,3,4,5],
               'learning_rate':[0.025,0.03,0.05,0.055],
               'min_child_weight':[2,3,4],
               'n_estimators':[10,20,50,100],
               'subsample':[0.1,0.2,0.4,0.6]
              }


# In[178]:


grid_xgb = GridSearchCV(estimator  = xgb,
                         param_grid = params_xgb,
                         scoring    = 'roc_auc',
                         cv         = 3,
                         n_jobs     = -1,
                         verbose    = True
                        )


# In[179]:


grid_xgb.fit(x_train, y_train)


# In[180]:


print("Best Hyperparameters: \n", grid_xgb.best_params_)
print("Best AUC Score: \n", grid_xgb.best_score_)


# In[181]:


# Save the best combination of parameters as best model
best_model_xgb = grid_xgb.best_estimator_


# In[182]:


y_test_pred = best_model_xgb.predict(x_test)
y_train_pred = best_model_xgb.predict(x_train)


print("========================== XG Boost Tuned - Model Report =========================")

print("===================== Model Accuracy ==============================")
print("Train Accuracy: %0.3f"%(best_model_xgb.score(x_train, y_train)))
print("Test Accuracy: %0.3f"%(best_model_xgb.score(x_test, y_test)))
print("=====================================================")

# Performance Metrices
# from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

print("===================== Model Performance Metrices ===========================")
print("Test Confusion Matrix: ")
print(confusion_matrix(y_test, y_test_pred))
print("============================")
print("Train Confusion Matrix: ")
print(confusion_matrix(y_train, y_train_pred))

print("=====================================================")
print("Test Classification Report: ")
print(classification_report(y_test, y_test_pred))
print("=====================================================")
print("Train Classification Report: ")
print(classification_report(y_train, y_train_pred))

y_test_pred_prob = best_model_xgb.predict_proba(x_test)[:,1]
y_train_pred_prob = best_model_xgb.predict_proba(x_train)[:,1]

print("===================== Model AUC Scores ===========================")
print("Train AUC Score: %0.3f"%(roc_auc_score(y_train, y_train_pred_prob)))
print("Test AUC Score: %0.3f"%(roc_auc_score(y_test, y_test_pred_prob)))
print("=====================================================")

print("===================== Model AUC Curve ===========================")
fpr, tpr, threshold = roc_curve(y_test, y_test_pred_prob)
fpr1, tpr1, threshold1 = roc_curve(y_train, y_train_pred_prob)

plt.figure(figsize=(10,5))
plt.title('XG Boost Tuned ROC Curve')
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr1, tpr1, label='XG Boost Tuned - Train')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='XG Boost Tuned - Test')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
plt.show()


# In[183]:


# A new experiment


# In[186]:


with open('xgb_pkl_2', 'wb') as files:
    pickle.dump(best_model_xgb, files)


# In[185]:


import pickle


# ### Saving the best model as pickle model

# In[155]:


import pickle


# In[156]:


# xgboost and random forest


# In[157]:


with open('lgbm_pkl_1', 'wb') as files:
    pickle.dump(best_model_lgbm, files)


# In[158]:


# load saved model
with open('lgbm_pkl_1' , 'rb') as f:
    lgbm = pickle.load(f)


# In[159]:


with open('rf_pkl_1', 'wb') as files:
    pickle.dump(best_model_rf, files)


# In[160]:


# load saved model
with open('rf_pkl_1' , 'rb') as f:
    rf = pickle.load(f)


# In[235]:


with open('xgb_pkl_1', 'wb') as files:
    pickle.dump(best_model_xgb, files)


# In[236]:


# load saved model
with open('xgb_pkl_1' , 'rb') as f:
    xgb = pickle.load(f)


# # Step 5: Model Scoring & Business KPIs
# ### We will score the model with Validation Data
# 
# - Perform all the pre-processing steps like imputation, scaling etc using numbers from development sample
# - Score the model and report the Performance KPIs

# In[166]:


val=health_validation


# In[163]:


raw_features=pd.read_csv("C:\\Users\\Sahil\\Desktop\\Summer\\Project\\RAW_Features.csv")


# In[164]:


raw_feat=list(raw_features['Raw Features'])
raw_feat


# In[167]:


# Keeping only raw features
val_score = val[raw_feat]


# In[168]:


val_score.head(5)


# In[169]:


val_score['derived_ClmDiagnosisCode_9_flag'] = np.where(val_score['ClmDiagnosisCode_9_in_count']>val_score['ClmDiagnosisCode_9_in_nunique'], 1, 0)
val_score['derived_Race_3_flag'] = np.where(val_score['Race_3_merged_sum']>=30, 1, 0)


# In[194]:


raw_to_woe_feat=list(set(raw_feat)-set(['ClmDiagnosisCode_9_in_count'])-set(['ClmDiagnosisCode_9_in_nunique'])-set(['Provider'])-set(['PotentialFraud']))
raw_to_woe_feat


# In[195]:


dev_woe.head()


# In[196]:


# now to apply woebins on validation sample
#val_score_woe = sc.woebin_ply(val_score_dropped_provider, dev_woe)


# In[197]:


dev_woe_validation=pd.DataFrame()
for i in raw_to_woe_feat:
    dev_woe_validation=dev_woe_validation.append(bins[i])


# In[198]:


dev_woe_validation


# In[204]:


dev_woe_validation.to_csv("C:\\Users\\Sahil\\Desktop\\Summer\\Project\\dev_woe_validation.csv",index=False)


# In[205]:


dev_woe_validation=pd.read_csv("C:\\Users\\Sahil\\Desktop\\Summer\\Project\\dev_woe_validation.csv")


# In[ ]:





# In[176]:


df=pd.DataFrame()
test_df=pd.DataFrame(val_score[['Provider','ClmDiagnosisCode_7_out_nunique']])

test_woe_bin=bins['ClmDiagnosisCode_7_out_nunique']
test_df_woe=sc.woebin_ply(test_df,test_woe_bin)
test_df_woe.to_csv("C:\\Users\\Sahil\\Desktop\\Summer\\Project\\ClmDiagnosisCode_7_out_nunique_woe.csv",index=False)


# In[177]:


test_df.head()


# In[178]:


test_woe_bin.head()


# In[179]:


val_score.ClmDiagnosisCode_7_out_nunique.shape


# In[ ]:





# In[ ]:





# In[180]:


test_df_woe.head()


# In[181]:


val_score[raw_to_woe_feat].head()


# In[ ]:





# In[199]:


raw_to_woe_feat


# In[ ]:





# In[200]:


# now to apply woebins on validation sample
val_score_woe = sc.woebin_ply(val_score[['Provider','Race_2_merged_sum',
 'ClmDiagnosisCode_9_out_count',
 'AttendingPhysician_out_nunique',
 'OperatingPhysician_out_nunique',
 'ChronicCond_KidneyDisease_merged_sum',
 'ClmDiagnosisCode_5_in_count','InscClaimAmtReimbursed_out_mean',
 'ClmDiagnosisCode_6_out_count',
 'ClmDiagnosisCode_7_in_nunique',
 'OtherPhysician_in_nunique',
 'Claim_duration_merged_mean',
 'Race_5_merged_sum',
 'InscClaimAmtReimbursed_in_sum','InscClaimAmtReimbursed_merged_sum','Age_merged_mean',
 'ChronicCond_Alzheimer_merged_sum',
 'InscClaimAmtReimbursed_merged_std',
 'ChronicCond_Cancer_merged_sum',
 'Race_3_merged_sum',
 'ClmDiagnosisCode_8_in_count',
 'Age_merged_min',
 'Age_merged_max',
 'OtherPhysician_out_nunique',
 'InscClaimAmtReimbursed_in_min',
 'InscClaimAmtReimbursed_out_max','InscClaimAmtReimbursed_in_max',
 'InscClaimAmtReimbursed_merged_max',
 'ChronicCond_ObstrPulmonary_merged_sum','InscClaimAmtReimbursed_out_std',
 'OperatingPhysician_in_nunique',
 'ChronicCond_rheumatoidarthritis_merged_sum','ChronicCond_stroke_merged_sum','PotentialFraud'
 ]], dev_woe_validation)


# In[185]:


#val_score_woe['ClmDiagnosisCode_7_out_nunique_woe']=test_df_woe


# In[201]:


val_score_woe.head()


# In[202]:


val_score_woe = sc.woebin_ply(val_score[raw_to_woe_feat],dev_woe_validation)


# In[191]:


val_score_woe.head()


# In[ ]:




