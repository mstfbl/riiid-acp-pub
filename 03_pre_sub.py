#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.basics           import *
from fastai.callback.all     import *
from fastai.distributed      import *
from fastai.tabular.all      import *

import gc
import pandas as pd
import pickle

from pathlib import Path

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[2]:


in_d = Path('input')


# In[3]:


class _H:
    '''Hyperparams'''
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def __repr__(self):
        return str(self.__dict__)


# In[4]:


H = _H(
    chunk_size = 500, # trafo seq len
    data = '210101b',  # data version
)


# # Open pickles

# In[5]:


get_ipython().run_cell_magic('time', '', "with open(in_d / f'data_v{H.data}.pkl', 'rb') as f:\n    data = pickle.load(f)")


# # Get last `H.chunk_size-1` interactions of each user

# In[6]:


def tail_seqs(d):
    for k in d.keys():
        d[k] = d[k][-H.chunk_size+1:]


# In[7]:


tail_seqs(data.cat_d)
tail_seqs(data.cont_d)
tail_seqs(data.tags_d)
tail_seqs(data.tagw_d)


# # Sparse matrices to np

# In[8]:


attempt_num = data.attempt_num_coo.toarray()
attempts_correct = data.attempts_correct_coo.toarray()


# In[9]:


del data.attempt_num_coo
del data.attempts_correct_coo


# In[10]:


gc.collect()


# # Save pkl

# In[11]:


with open(in_d / f'data_{H.chunk_size}_last_interactions_v{H.data}.pkl', 'wb') as f:
    pickle.dump(data, f)


# # Save npy

# In[12]:


assert attempt_num.dtype == attempts_correct.dtype == np.uint8


# In[13]:


np.save(in_d / f'data_attempt_num_v{H.data}', attempt_num)


# In[14]:


np.save(in_d / f'data_attempts_correct_v{H.data}', attempts_correct)


# In[ ]:




