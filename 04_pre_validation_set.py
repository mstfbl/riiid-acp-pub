#!/usr/bin/env python
# coding: utf-8

# ```
# convert train.csv to a format suitable for inference
# ```

# In[1]:


import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# In[2]:


from fastai.basics           import *
from fastai.callback.all     import *
from fastai.distributed      import *
from fastai.tabular.all      import *

import ast
import enum
import gc
import pandas as pd
import pickle
import enum

from collections import defaultdict
from fastcore.script import *
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.sparse import coo_matrix, dok_matrix, lil_matrix, csr_matrix, bsr_matrix
from sklearn.metrics import roc_auc_score
from torch.distributions.beta import Beta
from torch.utils.data import Dataset
from tqdm import tqdm


# In[3]:


in_d = Path('input')


# In[4]:


MAX_USERS = 450000     # estimate users (train + test)
MAX_QUESTIONS = 13523  # num of different question_id


# In[5]:


_H = AttrDict
H = _H(
    seed = 0,
    data = '210101b',  # data version
    valid_pct = 0.025,
)


# In[6]:


with open(in_d / f'meta_v{H.data}.pkl', 'rb') as f:
    meta = pickle.load(f)

QCols = enum.IntEnum('QCols', meta.qcols, start=0)
LCols = enum.IntEnum('LCols', meta.lcols, start=0)
Cats  = enum.IntEnum('Cats',  meta.cat_names, start=0)
Conts = enum.IntEnum('Conts', meta.cont_names, start=0)


# # Seed np

# In[7]:


np.random.seed(H.seed)


# # Read df

# In[8]:


get_ipython().run_cell_magic('time', '', "\ninteraction_dtypes = {\n    'row_id': 'int32',\n    'timestamp': 'int64',\n    'user_id': 'int32',\n    'content_id': 'int16',\n    'content_type_id': 'int8',\n    'task_container_id': 'int16',\n    'user_answer': 'int8',\n    'answered_correctly': 'int8',\n    'prior_question_elapsed_time': 'float32',\n    'prior_question_had_explanation': 'boolean'\n}\n\ni_df = pd.read_csv(\n    in_d / 'train.csv', \n    usecols=interaction_dtypes.keys(),\n    dtype=interaction_dtypes,\n    #nrows=10**6,\n)")


# In[9]:


group_keys = sorted(i_df.user_id.unique())


# In[10]:


# Last H.valid_pct is valid set
train_group_keys = group_keys[:int((1 - H.valid_pct) * len(group_keys))]
valid_group_keys = group_keys[int((1 - H.valid_pct) * len(group_keys)):]


# In[11]:


print(f'users: train={len(train_group_keys)}, valid={len(valid_group_keys)}')


# In[12]:


df = i_df[i_df.user_id.isin(valid_group_keys)]


# In[13]:


assert len(valid_group_keys) == len(df.user_id.unique())


# In[14]:


stat = df.groupby('user_id').agg({'timestamp': [np.min, np.max, len]})


# In[15]:


max_ts = df.timestamp.max()


# In[16]:


df.timestamp.hist()


# In[17]:


stat.sort_values(('timestamp',  'len'))


# # Build test csv data

# In[18]:


#df[df.user_id == 1957824471][-100:]


# In[19]:


pd.set_option('display.max_rows', 500)


# In[20]:


df = df.sort_values(['timestamp', 'row_id'])


# In[21]:


idx = df.groupby(['user_id', 'timestamp']).head(1).index


# In[22]:


df['group_num'] = 0


# In[23]:


df.loc[idx, 'group_num'] = 1


# In[24]:


df.group_num = df.groupby('user_id').group_num.transform(lambda x: np.cumsum(x) - 1)


# In[25]:


max_gn = df.group_num.max()
max_gn


# In[26]:


def random_shift_ts(g):
    return g.timestamp + np.random.randint(0, max_gn - g.group_num.max() + 1)


# In[27]:


df.group_num = df.groupby('user_id').group_num.transform(lambda x: x + np.random.randint(0, max_gn - x.max() + 1))


# In[28]:


df.group_num.value_counts().plot()


# In[29]:


df = df.sort_values(['group_num', 'row_id'])


# In[30]:


gac = df.groupby('group_num').answered_correctly.apply(list)


# In[31]:


gr = df.groupby('group_num').user_answer.apply(list)


# In[32]:


idx = df.groupby('group_num').head(1).index


# In[33]:


df.loc[idx[0],  'prior_group_answers_correct'] = '[]'
df.loc[idx[1:], 'prior_group_answers_correct'] = gac[:-1].values


# In[34]:


df.loc[idx[0],  'prior_group_responses'] = '[]'
df.loc[idx[1:], 'prior_group_responses'] = gr[:-1].values


# In[35]:


df[df.group_num == 0]


# In[36]:


df[df.group_num == 1]


# In[37]:


df[-100:]


# ## Test csv

# In[38]:


test_cols = ('row_id,group_num,timestamp,user_id,content_id,content_type_id,task_container_id,'
    'prior_question_elapsed_time,prior_question_had_explanation,prior_group_answers_correct,'
    'prior_group_responses'.split(','))
test_cols


# In[39]:


df_test = df[test_cols]
df_test[:100]


# In[40]:


df_test.to_csv(in_d / f'validation_x_{H.valid_pct}.csv', index=False)


# ## Targets csv

# In[41]:


target_cols = ['row_id', 'answered_correctly', 'group_num']
target_cols


# In[42]:


df_target = df[df.answered_correctly != -1][target_cols]
df_target


# In[43]:


df_target.to_csv(in_d / f'validation_y_{H.valid_pct}.csv', index=False)


# ## Submission csv

# In[44]:


df_target.answered_correctly = 0.5


# In[45]:


df_target.to_csv(in_d / f'validation_submission_{H.valid_pct}.csv', index=False)


# In[ ]:




