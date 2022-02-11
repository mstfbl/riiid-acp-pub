#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
import joblib
import numpy as np
import pandas as pd
import pickle
import enum

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from scipy.sparse import coo_matrix, dok_matrix, lil_matrix, csr_matrix, bsr_matrix
from tqdm import tqdm


# # Input path
# Set this to the location of input .csv files

# In[2]:


in_d = Path('input')


# To line-profile a function:
# 
# ```
# conda install -c anaconda line_profiler
# ```
# 
# Then run:
# 
# ```
# %lprun -f f1 f2(...)
# ```
# 
# Both functions `f1` and `f2` can be the same or different ie. your code starts at `f2` but you are only interested in profiling `f1`
# 

# In[3]:


get_ipython().run_line_magic('load_ext', 'line_profiler')


# # Constants

# In[4]:


class _H:
    '''Hyperparams'''
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def __repr__(self):
        return str(self.__dict__)


# In[5]:


H = _H(
    version = '210101b',
    max_users = 450000,
    max_questions = 13523,
    valid_pct = 0.025, # ~2.5M rows
)


# # Helpers

# In[6]:


def categorize(df, cols):
    cats_d = {}
    for col in cols:
        if df[col].dtype.name == 'category':
            print(f'{col} already categorized')
        else:
            df[col] = pd.Categorical(df[col])
        cats_d[col] = df[col].cat.categories.values
    return cats_d


# In[7]:


def means_stds(df, cols):
    return { col: df[col].mean() for col in cols }, { col: df[col].std() for col in cols }


# # Questions

# In[8]:


question_dtypes = {
    'question_id': 'int16',
    'bundle_id': 'int16',
    'correct_answer': 'int8',
    'part': 'int8',
    'tags': 'object',
}

questions_df = pd.read_csv(
    in_d / 'questions.csv',
    usecols=question_dtypes.keys(),
    dtype=question_dtypes,
)

qcats = categorize(questions_df, ['question_id', 'bundle_id', 'correct_answer', 'part'])
qcats


# ## Split tags
# Tag `n` is renumbered to `n+1` so that 0 means "no tag"

# In[9]:


questions_df[[f'tag_{_}' for _ in range(6)]] = (questions_df.tags.str.split(expand=True).fillna('-1').astype('int16') + 1).astype('uint8')


# In[10]:


questions_df.info()


# In[11]:


questions_df


# In[12]:


questions_df = questions_df.drop('tags', axis=1)


# In[13]:


assert np.all(questions_df.isna() == False) # no nans
assert np.all(questions_df.values < 2**16) # all fit in int16


# In[14]:


qcols = questions_df.columns.to_list()
QCols = enum.IntEnum('QCols', qcols, start=0)


# In[15]:


qcodes_d = {}
for col, cats in qcats.items():
    # code=0 is reserved for <NA>, NaN and the likes
    qcodes_d[col] = { value: code+1 for code, value in enumerate(cats) }


# In[16]:


qc_d = {}
for row in questions_df.to_numpy():
    question_id_code = qcodes_d['question_id'][row[QCols.question_id]]
    qc_d[question_id_code] = np.array([
        question_id_code,
        qcodes_d['bundle_id'][row[QCols.bundle_id]],
        qcodes_d['correct_answer'][row[QCols.correct_answer]],
        qcodes_d['part'][row[QCols.part]],
        row[QCols.tag_0],
        row[QCols.tag_1],
        row[QCols.tag_2],
        row[QCols.tag_3],
        row[QCols.tag_4],
        row[QCols.tag_5],
    ], dtype=np.int16)


# # Lectures

# In[17]:


lecture_dtypes = {
    'lecture_id': 'int64',
    'tag': 'uint8',
    'part': 'int8',
    'type_of': 'object'
}

lectures_df = pd.read_csv(
    in_d / 'lectures.csv',
    usecols=lecture_dtypes.keys(),
    dtype=lecture_dtypes,
)

lcats = categorize(lectures_df, ['lecture_id', 'part', 'type_of'])
#lectures_cats


# In[18]:


assert np.all(lcats['part'] == qcats['part']) # all parts show up on both dfs


# ## Tags
# Tag `n` is renumbered to `n+1` here as well

# In[19]:


lectures_df['tag_0'] = (lectures_df.tag.fillna('-1').astype('int16') + 1).astype('uint8')
for i in range(1, 6):
    lectures_df[f'tag_{i}']= pd.Series(0, index=lectures_df.index, dtype='uint8')


# In[20]:


lectures_df.info()


# In[21]:


lectures_df


# In[22]:


lectures_df = lectures_df.drop('tag', axis=1)


# In[23]:


assert lectures_df.isna().sum().sum() == 0


# In[24]:


lcols = lectures_df.columns.to_list()
LCols = enum.IntEnum('LCols', lcols, start=0)


# In[25]:


lcodes_d = {}
for col, cats in lcats.items():
    # code=0 is reserved for <NA>, NaN and the likes
    lcodes_d[col] = { value: code+1 for code, value in enumerate(cats) }


# In[26]:


assert max([ max(col_codes.values()) for col_codes in lcodes_d.values() ]) < 2**15 # fit in int16?


# In[27]:


lc_d = {}
for row in lectures_df.to_numpy():
    lecture_id_code = lcodes_d['lecture_id'][row[LCols.lecture_id]]
    lc_d[lecture_id_code] = np.array([
        lecture_id_code,
        lcodes_d['part'][row[LCols.part]],
        lcodes_d['type_of'][row[LCols.type_of]],
        row[LCols.tag_0],
        row[LCols.tag_1],
        row[LCols.tag_2],
        row[LCols.tag_3],
        row[LCols.tag_4],
        row[LCols.tag_5],
    ], dtype=np.int16)


# # Interactions

# In[28]:


get_ipython().run_cell_magic('time', '', "interaction_dtypes = {\n    'row_id': 'int32',\n    'timestamp': 'int64',\n    'user_id': 'int32',\n    'content_id': 'int16',\n    'content_type_id': 'int8',\n    'task_container_id': 'int16',\n    'user_answer': 'int8',\n    'answered_correctly': 'int8',\n    'prior_question_elapsed_time': 'float32',\n    'prior_question_had_explanation': 'boolean'\n}\n\ni_df = pd.read_csv(\n    in_d / 'train.csv', \n    usecols=interaction_dtypes.keys(),\n    dtype=interaction_dtypes,\n    #nrows=10**6,\n)\n\nicats = categorize(i_df, ['task_container_id', 'user_answer', 'answered_correctly', 'prior_question_had_explanation'])")


# In[29]:


icats['prior_question_had_explanation']


# In[30]:


icols = i_df.columns.to_list()
ICols = enum.IntEnum('ICols', icols, start=0)


# In[31]:


icols


# In[32]:


icodes_d = {}
for col, cats in icats.items():
    # code=0 is reserved for <NA>, NaN and the likes
    icodes_d[col] = { value: code+1 for code, value in enumerate(cats) }


# In[33]:


# hack in <NA> in cats_d['prior_question_had_explanation']
icodes_d['prior_question_had_explanation'][pd.NA] = 0
icodes_d['prior_question_had_explanation'][np.nan] = 0


# In[34]:


max_icode = max([ max(col_codes.values()) for col_codes in icodes_d.values() ])
assert max_icode < 2**15 # fit in int16?


# ## Merge all codes into codes_d

# In[35]:


codes_d = { **icodes_d, **qcodes_d, **lcodes_d }


# In[36]:


cat_names = sorted([
    'already_answered',          # has this question been answered before?
    'answered_correctly',        # answered correctly by user
    'bundle_id', 
    'correct_answer', 
    'lecture_id', 
    'part', 
    'qhe',                       # question has explanation (pqhe shifted upwards 1 container)
    'question_id', 
    'task_container_id',
    'type_of',                   # lecture type
    'user_answer', 
])

# To hide from decoder:
# - answered_correctly
# - user_answer
# - qhe

cont_names = sorted([
    'attempt_num',               # number of attempts per user_id, question_id
    'attempt_num_log',           # log1p of the above
    'attempts_correct',          # number of CORRECT attempts per user_id, question_id
    'attempts_correct_log',
    'attempts_correct_avg',      # attempts_correct / attempts_num
    'attempts_correct_avg_log',
    'container_ord',             # ordinal of question within container
    'qet',                       # question elapsed time (pqet shifted upwards 1 container)
    'qet_log',
    'qp',                        # probabilty of occurrence of this question
    'qp_log',
    'timestamp',                 # interaction ts
    'timestamp_log',
    'tsli',                      # time since last interaction (aka timestamp delta)
    'tsli_log',
    'clipped_tsli',              # tsli clipped to 20 minutes
    'clipped_tsli_log',
    'ts_mod_1day',               # timestamp modulus 1 day
    'ts_mod_1day_sin',
    'ts_mod_1day_cos',
    'ts_mod_1week',              # timestamp modulus 1 week
    'ts_mod_1week_sin',
    'ts_mod_1week_cos',
])

# To hide from decoder:
# - qet
# - qet_log


# In[37]:


Cats = enum.IntEnum('Cats', cat_names, start=0)
Conts = enum.IntEnum('Conts', cont_names, start=0)


# ## Encode user_ids
# This helps coo -> lil_matrix conversion to not freak out

# In[38]:


users_d = defaultdict(lambda: len(users_d))
for user_id in np.sort(i_df.user_id.unique()):
    users_d[user_id]


# In[39]:


assert len(users_d.keys()) == 393656
assert np.all(np.array(list(users_d.keys())) == np.array(sorted(users_d.keys())))
assert users_d[2746] == 2


# ## Find probabilty of occurrence of each question

# In[40]:


get_ipython().run_cell_magic('time', '', 'tmp_q_df = i_df[i_df.content_type_id == 0]\nqp_d = (tmp_q_df.content_id.value_counts() / len(tmp_q_df)).to_dict()\ndel tmp_q_df')


# ## `update_questions`, `update_answers`, `get_x` 

# In[41]:


def update_questions(df, Col, cat_names, cont_names, qc_d, lc_d, codes_d, QCols, LCols, Cats, Conts, 
        hist_cat_d, hist_cont_d, hist_tags_d, hist_tagw_d, last_q_container_d, last_ts, attempt_num, 
        attempts_correct, qp_d, users_d):
    
    df_a = df.values
    
    n_rows = len(df)
    
    # Prefetch tslis per (user_id, tcid) for better_tsli calculation
    # NOTE the keys (user_id, tcid) are NOT encoded
    tsli_d = defaultdict(list)
    #for i, (_, row) in enumerate(df_d.items()): # SLOW
    #for i, (_, row) in enumerate(df.iterrows()): # SUPER SLOW
    for i, row in enumerate(df_a):
        user_id, tcid, ts = row[Col.user_id], row[Col.task_container_id], row[Col.timestamp]
        encoded_user_id = users_d[user_id]
        tsli_d[user_id, tcid].append(ts - last_ts[encoded_user_id,0])
        last_ts[encoded_user_id,0] = np.int64(ts)
        
    # average all tslis in the same task container
    tsli_d = { k: sum(v)/len(v) for k, v in tsli_d.items() }
    
    # append df data to history
    for i, row in enumerate(df_a):
        user_id = row[Col.user_id]
        encoded_user_id = users_d[user_id]
        user_has_hist = user_id in hist_cat_d
        if user_has_hist:
            h_cat  = hist_cat_d [user_id] # just shortcuts
            h_cont = hist_cont_d[user_id]
            h_tags = hist_tags_d[user_id]
            h_tagw = hist_tagw_d[user_id]
        
        cat  = np.zeros(len(cat_names),  dtype=np.int16)
        cont = np.full (len(cont_names), np.nan, dtype=np.float32)

        # Categorical test data
        content_id = row[Col.content_id]
        is_question = row[Col.content_type_id] == 0

        if is_question:
            encoded_question_id = codes_d['question_id'][content_id]
            qc_row = qc_d[encoded_question_id]
            cat[Cats.bundle_id]        = qc_row[QCols.bundle_id]
            cat[Cats.correct_answer]   = qc_row[QCols.correct_answer]
            cat[Cats.part]             = qc_row[QCols.part]
            cat[Cats.question_id]      = encoded_question_id
            cat[Cats.already_answered] = (int)(attempt_num[encoded_user_id, encoded_question_id-1] > 0)
            cat[Cats.qhe]              = 0  # question has explanation?, not known yet    
        else:
            encoded_lecture_id = codes_d['lecture_id'][content_id]
            lc_row = lc_d[encoded_lecture_id]
            cat[Cats.lecture_id] = encoded_lecture_id
            cat[Cats.part]       = lc_row[LCols.part]
            cat[Cats.type_of]    = lc_row[LCols.type_of]

        tcid = row[Col.task_container_id]
        encoded_pqhe = codes_d['prior_question_had_explanation'][row[Col.prior_question_had_explanation]]
        encoded_tcid = codes_d['task_container_id'][tcid]
        cat[Cats.task_container_id] = encoded_tcid
        
        # Continuous test data
        ts = row[Col.timestamp]
        ts_mod_1day = ts % (1000 * 60 * 60 * 24)
        ts_mod_1week = ts % (1000 * 60 * 60 * 24 * 7)
        pqet = row[Col.prior_question_elapsed_time]
        tsli = tsli_d[(user_id, tcid)] if user_has_hist else np.nan
        clipped_tsli = min(tsli, 1000 * 60 * 20) # 20 minutes
        
        cont[Conts.qet]              = np.nan
        cont[Conts.timestamp]        = ts
        cont[Conts.tsli]             = tsli
        cont[Conts.clipped_tsli]     = clipped_tsli
        cont[Conts.qet_log]          = np.nan
        cont[Conts.timestamp_log]    = np.log1p(ts)
        cont[Conts.tsli_log]         = np.log1p(tsli)
        cont[Conts.clipped_tsli_log] = np.log1p(clipped_tsli)
        cont[Conts.ts_mod_1day]      = ts_mod_1day
        cont[Conts.ts_mod_1day_sin]  = np.sin(ts_mod_1day * 2 * np.pi / (1000 * 60 * 60 * 24))
        cont[Conts.ts_mod_1day_cos]  = np.cos(ts_mod_1day * 2 * np.pi / (1000 * 60 * 60 * 24))
        cont[Conts.ts_mod_1week]     = ts_mod_1week
        cont[Conts.ts_mod_1week_sin] = np.sin(ts_mod_1week * 2 * np.pi / (1000 * 60 * 60 * 24 * 7))
        cont[Conts.ts_mod_1week_cos] = np.cos(ts_mod_1week * 2 * np.pi / (1000 * 60 * 60 * 24 * 7))
        
        # container ordinal
        if user_has_hist and h_cat[-1,Cats.task_container_id] == encoded_tcid:
            cont[Conts.container_ord] = h_cont[-1,Conts.container_ord] + 1
        else:
            cont[Conts.container_ord] = 0
        
        if is_question:
            # Update qet and qet_log in history (make qet in last bundle skipping lectures = pqet)
            if user_id in last_q_container_d and encoded_tcid != last_q_container_d[user_id]:
                idx = h_cat[:,Cats.task_container_id] == last_q_container_d[user_id]
                h_cat [idx,Cats.qhe]      = encoded_pqhe
                h_cont[idx,Conts.qet]     = pqet
                h_cont[idx,Conts.qet_log] = np.log1p(pqet)
                        
            last_q_container_d[user_id] = encoded_tcid
            
            # Update attempt_num
            an = attempt_num     [encoded_user_id, encoded_question_id-1] # np.uint8
            ac = attempts_correct[encoded_user_id, encoded_question_id-1] # np.uint8
            cont[Conts.attempt_num]              = an
            cont[Conts.attempt_num_log]          = np.log1p(an)
            
            # Update attempts_correct with what we know so far (will be re-updated after we've got the answers)
            cont[Conts.attempts_correct]         = ac
            cont[Conts.attempts_correct_log]     = np.log1p(ac)
            if an != 0:
                cont[Conts.attempts_correct_avg]     = ac / an
                cont[Conts.attempts_correct_avg_log] = np.log1p(ac / an)

            attempt_num[encoded_user_id, encoded_question_id-1] += np.uint8(1)

            # question occurrence prob
            cont[Conts.qp]              = qp_d[content_id] # qp_d indexes are non-encoded qids
            cont[Conts.qp_log]          = np.log1p(cont[Conts.qp])

        # Tags and weights
        if is_question:
            tags = qc_row[[ QCols.tag_0, QCols.tag_1, QCols.tag_2, QCols.tag_3, QCols.tag_4, QCols.tag_5 ]]
        else:
            tags = lc_row[[ LCols.tag_0, LCols.tag_1, LCols.tag_2, LCols.tag_3, LCols.tag_4, LCols.tag_5 ]]
        tags = tags.astype(np.uint8)
        tagw = (tags != 0).astype(np.float16)
        sums = tagw.sum()
        if sums > 0:
            tagw /= sums
       
        # Concat history and new test data
        if user_has_hist:
            hist_cat_d [user_id] = np.concatenate((h_cat,  np.expand_dims(cat,  0)))
            hist_cont_d[user_id] = np.concatenate((h_cont, np.expand_dims(cont, 0)))
            hist_tags_d[user_id] = np.concatenate((h_tags, np.expand_dims(tags, 0)))
            hist_tagw_d[user_id] = np.concatenate((h_tagw, np.expand_dims(tagw, 0)))
        else:
            hist_cat_d [user_id] = np.expand_dims(cat,  0)
            hist_cont_d[user_id] = np.expand_dims(cont, 0)
            hist_tags_d[user_id] = np.expand_dims(tags, 0)
            hist_tagw_d[user_id] = np.expand_dims(tagw, 0)

    return df.user_id.values


def update_answers(prior_user_ids, prior_group_answers_correct, prior_group_responses, 
        cat_names, cont_names, codes_d, hist_cat_d, hist_cont_d, users_d, attempt_num, attempts_correct):

    idx_per_uid_d = defaultdict(int)
    for uid in prior_user_ids:
        idx_per_uid_d[uid] -= 1

    for i, uid in enumerate(prior_user_ids):
        h_cat  = hist_cat_d [uid] # just shortcuts
        h_cont = hist_cont_d[uid]
        
        idx = idx_per_uid_d[uid]
        idx_per_uid_d[uid] += 1
        
        # Update categorical vars
        h_cat [idx,Cats.answered_correctly] = codes_d['answered_correctly'][prior_group_answers_correct[i]]
        h_cat [idx,Cats.user_answer]        = codes_d['user_answer'][prior_group_responses[i]]

        # Update continuous vars
        eqid = h_cat[idx,Cats.question_id]
        if eqid > 0: # it's a question
            assert prior_group_answers_correct[i] >= 0
            euid = users_d[uid]
            ac = attempts_correct[euid,eqid-1] # np.int8
            an = h_cont[idx,Conts.attempt_num] # np.float32
            h_cont[idx,Conts.attempts_correct]         = ac
            h_cont[idx,Conts.attempts_correct_log]     = np.log1p(ac)
            if an != 0:
                h_cont[idx,Conts.attempts_correct_avg]     = ac / an
                h_cont[idx,Conts.attempts_correct_avg_log] = np.log1p(ac / an)

            attempts_correct[euid,eqid-1] = ac + np.uint8(prior_group_answers_correct[i])
        else:
            assert prior_group_answers_correct[i] == -1


# ## ```proxy_append_df```

# In[42]:


def proxy_append_df(df):
    hist_cat_d         = {}
    hist_cont_d        = {}
    hist_tags_d        = {}
    hist_tagw_d        = {}
    last_q_container_d = {}
    last_ts            = defaultdict(np.int64)
    attempt_num        = defaultdict(np.uint8)
    attempts_correct   = defaultdict(np.uint8)
    chunk_size         = None
    Col                = enum.IntEnum('Col', df.columns.tolist(), start=0)

    # update questions
    prior_user_ids = update_questions(
        df, Col, cat_names, cont_names, qc_d, lc_d, codes_d, QCols, LCols, Cats, Conts, 
        hist_cat_d, hist_cont_d, hist_tags_d, hist_tagw_d, last_q_container_d, last_ts, 
        attempt_num, attempts_correct, qp_d, users_d)

    # update answers
    prior_group_answers_correct = df.answered_correctly.values
    prior_group_responses       = df.user_answer.values

    update_answers(prior_user_ids, prior_group_answers_correct, prior_group_responses, 
        cat_names, cont_names, codes_d, hist_cat_d, hist_cont_d, users_d, attempt_num, attempts_correct)
    
    return (hist_cat_d, hist_cont_d, hist_tags_d, hist_tagw_d, 
            last_q_container_d, last_ts, attempt_num, attempts_correct)


# In[ ]:





# ## Test `append_df`

# In[43]:


#%lprun -f update_answers (
#    hist_cat_d, hist_cont_d, hist_tags_d, hist_tagw_d, 
#    last_last_q_container_d, last_ts, attempt_num, previous_ac) = proxy_append_df(i_df[:100000])


# In[44]:


#%%time
#(hist_cat_d, hist_cont_d, hist_tags_d, hist_tagw_d,
# last_last_q_container_d, last_ts, attempt_num, attempts_correct) = proxy_append_df(i_df[:100000])


# * pytorch.sparse -> 32s and can't parallelize ("torch sparse tensor has no storage" error)
# * scipy.sparse.dok_matrix -> 17.5s
# * scipy.sparse.lil_matrix -> 15.4s
# * scipy.sparse.csr_matrix -> minutes

# In[45]:


#pd.set_option('display.max_rows', 1000)
#pd.set_option('display.max_columns', None)


# Interesting user_ids:
# 
# - 8623 (3 containers x 5 questions)
# - 124 (1 container, meaningful tsli)
# 

# In[46]:


#i_df[i_df.user_id == 8623]


# In[47]:


#pd.DataFrame(hist_ord_d[8623], columns=['ordinal'])


# In[48]:


#pd.DataFrame(hist_cat_d[8623], columns=cat_names)


# In[49]:


# just for the header
#pd.DataFrame(hist_cont_d[8623][:1], columns=cont_names)


# In[50]:


#test_df = pd.DataFrame(hist_cont_d[8623], columns=cont_names)


# In[51]:


#test_df


# In[52]:


#test_df[test_df.attempt_num > 1]


# In[53]:


#all_cont = np.concatenate(list(v for v in hist_cont_d.values()))


# In[54]:


#_=plt.hist(np.log1p(all_cont[:,Conts.tsli]), bins=100)


# In[55]:


#np.exp(14) / 1000 / 60


# In[56]:


#pd.DataFrame(hist_cat_d[124], columns=cat_names)


# In[57]:


#pd.DataFrame(hist_cont_d[115], columns=cont_names)


# # Create training data

# In[58]:


bins = np.linspace(i_df.user_id.min()-1, i_df.user_id.max(), num=1024+1, dtype=np.int32)


# In[59]:


dfg = i_df.groupby(pd.cut(i_df.user_id, bins))


# In[60]:


get_ipython().run_cell_magic('time', '', 'groups = [ dfg.get_group(_) for _ in dfg.groups.keys() ]')


# In[61]:


with ProcessPoolExecutor() as e:
    res = list(tqdm(e.map(proxy_append_df, groups), total=len(groups)))


# In[98]:


get_ipython().run_cell_magic('time', '', 'merge_dicts = lambda idx: { k: v for d in [ _[idx] for _ in res ] for k, v in d.items() }\ncat_d                 = merge_dicts(0)\ncont_d                = merge_dicts(1)\ntags_d                = merge_dicts(2)\ntagw_d                = merge_dicts(3)\nlast_q_container_id_d = merge_dicts(4)\nlast_ts               = merge_dicts(5)\nattempt_num           = merge_dicts(6)\nattempts_correct      = merge_dicts(7)')


# ## Dok matrices

# In[63]:


assert all(v.dtype == np.int64 for v in last_ts.values())
assert all(v.dtype == np.uint8 for v in attempt_num.values())
assert all(v.dtype == np.uint8 for v in attempts_correct.values())


# In[64]:


last_ts_dok          = dok_matrix((H.max_users, 1), dtype=np.int64)
attempt_num_dok      = dok_matrix((H.max_users, H.max_questions), dtype=np.uint8)
attempts_correct_dok = dok_matrix((H.max_users, H.max_questions), dtype=np.uint8)


# In[65]:


last_ts_dok._update(last_ts)


# In[66]:


attempt_num_dok._update(attempt_num)


# In[67]:


attempts_correct_dok._update(attempts_correct)


# In[68]:


#del res  # this barely has an effect
#gc.collect()


# ## Convert dok -> array or coo
# 

# In[69]:


last_ts_dok


# In[70]:


get_ipython().run_cell_magic('time', '', 'last_ts = last_ts_dok.toarray()')


# In[71]:


attempt_num_dok


# In[72]:


get_ipython().run_cell_magic('time', '', 'attempt_num_coo = attempt_num_dok.tocoo()')


# In[73]:


attempts_correct_dok


# In[74]:


get_ipython().run_cell_magic('time', '', 'attempts_correct_coo = attempts_correct_dok.tocoo()')


# In[75]:


del attempt_num, attempts_correct
gc.collect()


# In[76]:


assert len(last_q_container_id_d) == len(cat_d) == len(i_df.user_id.unique()) 


# In[77]:


assert attempt_num_coo.getnnz() == 86867031


# In[78]:


test_ts = i_df.groupby('user_id')['timestamp'].max().values
np.testing.assert_equal(test_ts, last_ts[:len(test_ts),0])


# # Embedding sizes

# In[79]:


n_emb = {
    'already_answered': 2,
    'answered_correctly': 4,
    'bundle_id': 9766,
    'correct_answer': 5,
    'lecture_id': 419,
    'part': 8,
    'prior_question_had_explanation': 3,
    'question_id': 13524,
    'task_container_id': 10001,
    'type_of': 5,
    'user_answer': 6
}

emb_dim = {
    'already_answered': 1,
    'answered_correctly': 3,
    'bundle_id': 274,
    'correct_answer': 4,
    'lecture_id': 47,
    'part': 5,
    'prior_question_had_explanation': 3,
    'question_id': 329,
    'task_container_id': 278,
    'type_of': 4,
    'user_answer': 4
}

tags_n_emb = 187+2 # [0..max_tag, max_tag+1]. max_tag+1 = empty tag 
tags_emb_dim = tags_n_emb # emb_sz_rule(tags_n_emb)


# # Some checks

# In[80]:


all_cat = np.concatenate(list(cat_d.values()))
assert np.isnan(all_cat).sum() == 0


# In[81]:


assert tags_d[115].dtype == np.uint8


# # Means and stds of continuous vars

# In[82]:


all_cont = np.concatenate(list(cont_d.values()))
assert all_cont.shape[0] == len(i_df)


# In[83]:


assert np.nanmax(all_cont[:,Conts.attempt_num]) == 82


# In[84]:


#assert np.isnan(all_cont[:,Conts.prior_question_elapsed_time]).sum() == i_df.prior_question_elapsed_time.isna().sum()


# In[85]:


means = np.nanmean(all_cont, axis=0, dtype=np.float64)
stds  = np.nanstd (all_cont, axis=0, dtype=np.float64)


# In[86]:


maxs = np.nanmax(all_cont, axis=0)
mins = np.nanmin(all_cont, axis=0)


# In[87]:


cont_names, means, stds, maxs, mins


# In[88]:


means[Conts.tsli]


# In[89]:


assert np.abs(means[Conts.timestamp] - 7703643654.326523) < 0.1 # vs 52
#assert np.abs(means[Conts.prior_question_elapsed_time] - 25423.844) < 0.1 # vs 52
assert np.abs(means[Conts.tsli] - 20061832.08) < 50 # vs 201221 (precise timestamp)
assert np.abs(means[Conts.qet] - 2.59525175e+04) < 0.1 # vs 62, TODO: fix bundle_id -> tcid in 52 and get baseline


# In[90]:


#pd.set_option('display.max_rows', 128)


# In[91]:


#pd.DataFrame(all_cont[:100], columns=cont_names)


# In[92]:


#pd.DataFrame(all_cat[:100], columns=cat_names)


# # Pickle train data

# WTF, can't pickle enums... We'll rebuild them at train/infer
# ```
# QCols = enum.IntEnum('QCols', meta.qcols, start=0)
# LCols = enum.IntEnum('LCols', meta.lcols, start=0)
# Cats  = enum.IntEnum('Cats',  meta.cat_names, start=0)
# Conts = enum.IntEnum('Conts', meta.cont_names, start=0)
# ```

# In[93]:


meta = _H(
    means=means,
    stds=stds,
    maxs=maxs,
    mins=mins,
    qc_d=qc_d,
    qcats=qcats,
    qcols=qcols,
    qcodes_d=qcodes_d,
    lc_d=lc_d,
    lcats=lcats,
    lcols=lcols,
    lcodes_d=lcodes_d,
    codes_d=codes_d,
    cat_names=cat_names,
    cont_names=cont_names,
    icats=icats,
    n_emb=n_emb,
    emb_dim=emb_dim,
    tags_n_emb=tags_n_emb,
    tags_emb_dim=tags_emb_dim,
)


# In[94]:


data = _H(
    cat_d=cat_d,
    cont_d=cont_d,
    tags_d=tags_d,
    tagw_d=tagw_d,
    last_q_contained_id_d=last_q_container_id_d,
    attempt_num_coo=attempt_num_coo,
    attempts_correct_coo=attempts_correct_coo,
    last_ts=last_ts,
    qp_d=qp_d,
)


# In[95]:


H.version


# In[96]:


get_ipython().run_cell_magic('time', '', "with open(in_d / f'data_v{H.version}.pkl', 'wb') as f:\n    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)")


# In[97]:


get_ipython().run_cell_magic('time', '', "with open(in_d / f'meta_v{H.version}.pkl', 'wb') as f:\n    pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)")

