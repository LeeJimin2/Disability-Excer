#!/usr/bin/env python
# coding: utf-8

# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


import pandas as pd
import os
import numpy as np


# 체력측정데이터셋 생성

# In[ ]:


path = "/content/drive/MyDrive/경진대회/체력측정데이터/"
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.csv')] 
file_list_py


# In[ ]:


## 각 csv 불러와서 concat으로 한 데이터 프레임 연걸
mse_df = pd.DataFrame()
for i in file_list_py:
    data = pd.read_csv(path + i,encoding = 'UTF-8')
    mse_df = pd.concat([mse_df,data])
mse_df = mse_df.reset_index(drop = True)


# 운동처방 데이터셋 생성

# In[ ]:


path = "/content/drive/MyDrive/경진대회/운동처방/"
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.csv')] 
file_list_py


# In[ ]:


prs_df = pd.DataFrame()
for i in file_list_py:
    data_prs = pd.read_csv(path + i,encoding = 'UTF-8')
    prs_df = pd.concat([prs_df,data_prs])


prs_df = prs_df.reset_index(drop = True)


# In[ ]:


# =============================================================================
# 나이 범주화
# =============================================================================
def age_categorize(age):
    age = (age//10) * 10
    return age


# In[ ]:


### 체력측정데이터 적용
mse_df.MESURE_AGE_CO = mse_df.MESURE_AGE_CO.apply(age_categorize)
mse_df.MESURE_AGE_CO = mse_df.MESURE_AGE_CO .astype('category')


# In[ ]:


### 운동처방데이터 적용
prs_df.MESURE_AGE_CO = prs_df.MESURE_AGE_CO.apply(age_categorize)
prs_df.MESURE_AGE_CO = prs_df.MESURE_AGE_CO .astype('category')


# In[ ]:


mse_df.isna().sum()


# 지적장애 운동 측정 데이터

# In[ ]:


data_df.rename(columns = {'MESURE_IEM_001_VALUE' : '수축기혈압(최고)mmHg',
                       'MESURE_IEM_002_VALUE' : '이완기혈압(최저)mmHg',
                       'MESURE_IEM_003_VALUE' : '신장(cm)',
                       'MESURE_IEM_004_VALUE' : '체중(kg)',
                       'MESURE_IEM_005_VALUE' : 'BMI',
                       'MESURE_IEM_006_VALUE' : '체지방율',
                       'MESURE_IEM_007_VALUE' : '피부두겹1_삼두근',
                       'MESURE_IEM_008_VALUE' : '피부두겹1_이두근',
                       'MESURE_IEM_009_VALUE' : '피부두겹1_견갑하부',
                       'MESURE_IEM_010_VALUE' : '피부두겹1_장골능',
                       'MESURE_IEM_011_VALUE' : '피부두겹2_가슴',
                       'MESURE_IEM_012_VALUE' : '피부두겹2_복부',
                       'MESURE_IEM_013_VALUE' : '피부두겹2_대퇴',
                       'MESURE_IEM_014_VALUE' : '피부두겹2_삼두근',
                       'MESURE_IEM_015_VALUE' : '피부두겹2_상장골',
                       'MESURE_IEM_016_VALUE' : '악력',
                       'MESURE_IEM_017_VALUE' : '악력좌',
                       'MESURE_IEM_018_VALUE' : '악력우',
                       'MESURE_IEM_019_VALUE' : '암컬',
                       'MESURE_IEM_020_VALUE' : '암컬좌',
                       'MESURE_IEM_021_VALUE' : '암컬우',
                       'MESURE_IEM_022_VALUE' : '윗몸일으키기',
                       'MESURE_IEM_023_VALUE' : '5분달리기(휠체어)',
                       'MESURE_IEM_024_VALUE' : '암에르고미터',
                       'MESURE_IEM_025_VALUE' : '6분걷기',
                       'MESURE_IEM_026_VALUE' : '스텝검사',
                       'MESURE_IEM_027_VALUE' : '스텝검사회복시심박수',
                       'MESURE_IEM_028_VALUE' : '페이서',
                       'MESURE_IEM_029_VALUE' : '등뒤로손잡기',
                       'MESURE_IEM_030_VALUE' : '윗몸앞으로굽히기'}, inplace = True)


# In[3]:


data_df = pd.read_csv('/content/drive/MyDrive/경진대회/df (1).csv',encoding = 'UTF-8')
data_df.head()


# 결측치 처리하기

# In[ ]:


data_df.drop(columns = ['TROBL_DETAIL_NM','피부두겹1_삼두근','피부두겹1_이두근',
                        '피부두겹1_견갑하부','피부두겹1_장골능','암컬','암컬좌','암컬우',
                        '5분달리기(휠체어)','암에르고미터','페이서','등뒤로손잡기'],inplace = True)


# In[ ]:


data_df = data_df.reset_index(drop = True)
data_df


# In[ ]:


data_df = data_df.astype({'수축기혈압(최고)mmHg' : 'float64',
                       '이완기혈압(최저)mmHg' : 'float64',
                       '신장(cm)' : 'float64',
                       '체중(kg)' : 'float64',
                       '체지방율' : 'float64',
                       '악력' : 'float64',
                       '악력좌' : 'float64',
                       '악력우' : 'float64',
                       '윗몸일으키기' : 'float64',
                       '6분걷기' : 'float64',
                       '스텝검사회복시심박수' : 'float64',
                       '윗몸앞으로굽히기' : 'float64'})


# In[ ]:


data_df = data_df.fillna(method = 'ffill')
data_df


# In[ ]:


data_df = data_df.fillna(method = 'bfill')
data_df


# In[ ]:


data_df = data_df.fillna(data_df.interpolate())
data_df


# In[ ]:


data_df.info()


# In[ ]:


data_df.drop(columns = ['MESURE_TME','CNTER_NM','AGE_FLAG_NM',
                        'MESURE_PLACE_FLAG_NM','INPT_FLAG_NM','MESURE_DE',
                        'TROBL_TY_NM'],inplace = True)


# In[5]:


data_df.describe()


# In[4]:


data_df = pd.read_csv('/content/drive/MyDrive/경진대회/MSE_data.csv',encoding = 'UTF-8')
data_df


# In[ ]:


mse_prs_df = pd.merge(data_df,prs_df,how = 'inner')
mse_prs_df.drop_duplicates(inplace = True)
mse_prs_df.reset_index()


# In[ ]:


mse_prs_df.info()


# In[ ]:


mse_prs_df.to_csv('all.csv',encoding = 'UTF-8',index = False)


# 운동 처방 데이터셋
# 

# In[ ]:


prs_df = pd.read_csv('/content/drive/MyDrive/경진대회/Prscript.csv',encoding = 'UTF-8')


# In[ ]:


prs = prs_df[prs_df['TROBL_TY_NM'] == '지적장애']


# In[ ]:


prs.drop(columns = ['MESURE_TME','CNTER_NM','AGE_FLAG_NM',
                        'MESURE_PLACE_FLAG_NM','INPT_FLAG_NM','MESURE_DE',
                        'TROBL_TY_NM','TROBL_DETAIL_NM'],inplace = True)


# In[ ]:


prs = prs.reset_index(drop = True)
prs


# In[ ]:


prs_df = pd.concat([prs,arr_name],axis=1)


# In[5]:


prs_df = pd.read_csv('/content/drive/MyDrive/경진대회/prs_df.csv',encoding = 'CP949')
prs_df


# In[6]:


prs_df['MESURE_AGE_CO'] = prs_df['MESURE_AGE_CO'].replace(80,70)


# In[7]:


prs_df


# 추천 운동 데이터 

# In[8]:


recmd = pd.read_csv('/content/drive/MyDrive/경진대회/추천운동/KS_DSPSN_FTNESS_MESURE_ACCTO_RECOMEND_MVM_INFO_202205.csv',encoding = 'UTF-8')


# In[9]:


recmd = recmd[recmd['TROBL_TY_NM'] == '지적장애']


# In[10]:


recmd['SPORTS_STEP_NM'] = recmd['SPORTS_STEP_NM'].str.replace('준비운동','사전운동')


# In[11]:


recmd.drop(columns = ['TROBL_DETAIL_NM'],inplace = True)
recmd


# In[12]:


recmd = recmd.reset_index(drop = True)
recmd


# In[16]:


prs_df


# In[42]:


ABC_merge = ABC_merge.astype({'MESURE_AGE_CO' : 'int64',
                       'FLAG_ACCTO_RECOMEND_MVM_RANK_CO' : 'object',
                       })


# In[51]:


ABC_merge.to_csv('prs_score.csv',index = False, encoding = 'UTF-8')


# In[4]:


prs_score = pd.read_csv('/content/drive/MyDrive/경진대회/prs_score.csv',encoding = 'UTF-8')
prs_score


# In[116]:


prs_score.info()


# In[117]:


prs_score['마무리운동점수'] = prs_score['마무리운동점수'].fillna(0).astype('int8')


# In[118]:


prs_score = prs_score.astype({'MESURE_AGE_CO' : 'int64',
                       '사전운동점수' : 'int64',
                       '마무리운동점수' : 'int64'
                       })


# In[119]:


prs_score


# In[97]:


prs_score.info()


# In[88]:


prs_score['마무리운동점수'].replace([1,2,3,4,5],[5,4,3,2,1],inplace = True)


# In[89]:


prs_score


# In[101]:


score = []
for i in range(len(prs_score)):
  score.append((prs_score['사전운동점수'][i] + prs_score['본운동점수'][i] + prs_score['마무리운동점수'][i]) / 2)


# In[ ]:


score


# In[103]:


prs_score['score'] = score


# In[104]:


prs_score


# In[105]:


prs_score.drop(columns = ['사전운동점수','본운동점수','마무리운동점수'],inplace = True)
prs_score


# In[106]:


prs_score.to_csv('Score.csv',encoding = 'UTF-8',index = False)


# In[5]:


prs_score


# In[6]:


score_3 = prs_score.drop(columns = ['사전운동','사전운동점수','본운동','본운동점수'])


# In[8]:


score_3.dropna(inplace = True)


# In[9]:


score_3.to_csv('마무리운동.csv',encoding = 'UTF-8',index = False)


# Recommendataion System

# In[1]:


score = pd.read_csv('/content/drive/MyDrive/경진대회/Score.csv',encoding = 'UTF-8')
score.head()


# In[ ]:


data_df = pd.read_csv('/content/drive/MyDrive/경진대회/MSE_data.csv',encoding = 'UTF-8')
data_df


# In[108]:


data_df.head()


# In[111]:


score.shape


# In[112]:


data_df.shape


# In[1]:


score_3


# In[ ]:


score_1 = pd.read_csv('사전운동.csv',encoding = 'UTF-8')
score_2 = pd.read_csv('본운동.csv',encoding = 'UTF-8')
score_3 = pd.read_csv('마무리운동.csv',encoding = 'UTF-8')


# In[136]:


score_1


# In[4]:


data_df.descreibe


# In[ ]:


pd.merge(data_df,score_1,on = 'TROBL_GRAD_NM')


# In[128]:


score_1_pt = score_1.pivot_table('사전운동점수',index = 'TROBL_GRAD_NM',columns = '사전운동').fillna(0)
score_1_pt


# In[132]:


score_1_rating = score_1_pt.values.T


# In[129]:


score_2_pt = score_2.pivot_table('본운동점수',index = 'TROBL_GRAD_NM',columns = '본운동').fillna(0)
score_2_pt


# In[133]:


score_2_rating = score_2_pt.values.T


# In[130]:


score_3_pt = score_3.pivot_table('마무리운동점수',index = 'TROBL_GRAD_NM',columns = '마무리운동').fillna(0)
score_3_pt


# In[134]:


score_3_rating = score_3_pt.values.T

