#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#####운동 추천 tf-idf, 코사인유사도#####


# In[1]:


import pandas as pd


# In[2]:


연습 = pd.read_csv("연습.csv")


# In[3]:


연습 = 연습.loc[0]


# In[4]:


연습


# In[5]:


사전 = pd.read_csv("사전운동 (1).csv")


# In[6]:


본 = pd.read_csv("본운동 (1).csv")


# In[7]:


마무리 = pd.read_csv("마무리운동 (1).csv")


# In[8]:


사전.head()


# In[53]:


본.head()


# In[54]:


마무리.head()


# In[ ]:


##input으로 받은 데이터값을 sum과 같은 하나의 문장으로 받아서 cosine유사도 검사로 운동 추천


# In[9]:


cols = ['MESURE_AGE_CO','SEXDSTN_FLAG_CD','TROBL_GRAD_NM','수축기혈압(최고)mmHg','이완기혈압(최저)mmHg','신장(cm)','체중(kg)','BMI','체지방율','피부두겹2_가슴','피부두겹2_복부','피부두겹2_대퇴','피부두겹2_삼두근','피부두겹2_상장골','악력','악력좌','악력우','윗몸일으키기','6분걷기','스텝검사','스텝검사회복시심박수','윗몸앞으로굽히기']


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer


# In[11]:


vect = CountVectorizer()


# In[12]:


사전['sum'] = 사전[cols].apply(lambda row: ','.join(row.values.astype(str)), axis=1)


# In[13]:


본['sum'] = 본[cols].apply(lambda row: ','.join(row.values.astype(str)), axis=1)


# In[14]:


마무리['sum'] = 마무리[cols].apply(lambda row: ','.join(row.values.astype(str)), axis=1)


# In[65]:


마무리_df= 마무리["sum"]


# In[66]:


마무리_df


# In[67]:


마무리_df = 마무리_df.apply(str)


# In[68]:


마무리_df = 마무리_df.to_frame(name = "마무리")


# In[69]:


마무리_df = 마무리_df.head(20000)


# In[70]:


마무리_df


# In[71]:


마무리_df.loc[0] = ["30,M,3등급,157,88,173.8,91.5,30.3,34.6,18,105,10,85,5.9,32.1,31.1,32.1,22,66,39.8,111,4"]


# In[72]:


마무리_df


# In[25]:


사전_df


# In[26]:


import numpy as np


# In[27]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[73]:


tfidf_vect = TfidfVectorizer()


# In[74]:


tfidf_matrix = tfidf_vect.fit_transform(마무리_df['마무리']).todense()


# In[75]:


tfidf_matrix.shape


# In[57]:


tfidf_matrix[1]


# In[31]:


from sklearn.metrics.pairwise import cosine_similarity


# In[76]:


cosine_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[33]:


len(cosine_matrix)


# In[59]:


cosine_matrix


# In[77]:


idx2content = {}
for i, c in enumerate(마무리_df['마무리']): idx2content[i] = c
    
content2idx = {}
for i, c in idx2content.items(): content2idx[c] = i


# In[79]:


idx2content


# In[78]:


idx = 0


# In[46]:


content2idx


# In[80]:


sim_scores = [(i,c) for i,c in enumerate(cosine_matrix[idx]) if i != idx]
sim_scores = sorted(sim_scores, key = lambda x:x[1], reverse = True)
sim_scores[:3]


# In[81]:


sim_scores = [(idx2content[i], c) for i,c in sim_scores[:3]]
sim_scores


# In[50]:


idx2content[0]


# In[44]:


print("추천하는 사전 운동은 : ", 사전['사전운동'][1],",", 사전['사전운동'][2],",", 사전['사전운동'][3])


# In[64]:


print("추천하는 본 운동은 : ", 본['본운동'][1],",", 본['본운동'][2],",", 본['본운동'][3])


# In[82]:


print("추천하는 마무리 운동은 : ", 마무리['마무리운동'][19120],",", 마무리['마무리운동'][19121],",", 마무리['마무리운동'][19122])


# In[ ]:




