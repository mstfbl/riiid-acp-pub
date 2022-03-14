# %%
"""
To convert py from/to ipynb:

```
conda install conda install -c defaults -c conda-forge ipynb-py-convert
```

```
ipynb-py-convert script.py script.ipynb
ipynb-py-convert script.ipynb script.py
```
"""

# %%
# from fastai.basics           import *
# from fastai.callback.all     import *
# from fastai.distributed      import *
# from fastai.tabular.all      import *
from attrdict import AttrDict
import numpy as np
import torch
import torch.nn as nn
import re

import enum
import gc
import pandas as pd
import pickle
import time
import ast

from collections import defaultdict
from pathlib import Path
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

start_time = time.time()

# %%
start_time = time.time()

# %%
Mode = enum.IntEnum('Mode', ['normal', 'hurry_up', 'blindfolded_gunslinger'])

# %%
DEVICE              = 'cuda'
DO_NOT_UNTAR        = False
MODE                = Mode.normal
HURRY_UP_CUTOFF     = 0.25
BLIND_CUTOFF        = 0.19
PUB_PVT_CUTOFF      = 0.20
TIME_BUDGET         = 8.75 * 60 * 60 # secs

if MODE == Mode.normal or MODE == Mode.hurry_up:
    ROWS_TO_INFER = 2.5e6
elif MODE == Mode.blindfolded_gunslinger:
    ROWS_TO_INFER = (1-BLIND_CUTOFF) * 2.5e6

# %%
class _H:
    '''Hyperparams'''
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def __repr__(self):
        return str(self.__dict__)

# %%
H = AttrDict(
    {
        'chunk_size': 500, 
        'bs': 64,
        'valid_pct': 0.025,
        'data': '210101b',
    }
)
H1 = AttrDict(
    {
        'load': '210105_0.812534_relu_e3e3.pth',
        'trf_dim': 512,
        'trf_enc': 3,
        'trf_dec': 3,
        'trf_heads': 4, 
        'trf_do': 0.1, 
        'trf_act': 'relu', 
        'emb_do': 0.25, 
        'tfixup': True, 
    }
)

H2 = AttrDict(
    {
        'load': '210105_0.812154_gelu_e4d4_ep30.pth',
        'data': '210101',
        'trf_dim': 512,
        'trf_enc': 4,
        'trf_dec': 4,
        'trf_heads': 4, 
        'trf_do': 0.1, 
        'trf_act': 'gelu', 
        'emb_do': 0.25, 
        'tfixup': True, 
    }
)


# %%
KAGGLE = Path('/kaggle').exists()
KAGGLE

# %%
"""
# Env dependent paths
"""

# %%
if KAGGLE:
    # Use kaggle test sets and force GPU + untar resources
    ds_dir       = Path('/kaggle/input/riiid-acp')
    DEVICE       = 'cuda'
    DO_NOT_UNTAR = False
else:
    ds_dir       = Path('kaggle_dataset/to_upload')

# %%
in_d = Path('kaggle_dataset/root/resources')

# %%
"""
# Unpack dataset
"""

# %%
if not DO_NOT_UNTAR:
    if Path('resources').exists():
        shutil.rmtree('resources')
        
    for tgz in ds_dir.glob('*.tgz'):
        tgz = ds_dir / 'resources.tgz'
        assert os.system(f'tar xvf {str(tgz)}') == 0

# %%
"""
# Load data
"""

# %%
with open(in_d / f'data_{H.chunk_size}_last_interactions_v{H.data}.pkl', 'rb') as f:
    data = pickle.load(f)

# %%
# fix typo
data.last_q_container_id_d = data.last_q_contained_id_d

# %%
attempt_num = np.lib.format.open_memmap(in_d / f'data_attempt_num_v{H.data}.npy')
attempts_correct = np.lib.format.open_memmap(in_d / f'data_attempts_correct_v{H.data}.npy')

# %%
users_list = sorted(data.cat_d.keys())
users_d = defaultdict(lambda: len(users_d))
for user_id in users_list:
    users_d[user_id]
assert len(users_d.keys()) == 393656
assert users_d[2746] == 2
assert users_d[2126571790] == 389719

# %%
with open(in_d / f'meta_v{H.data}.pkl', 'rb') as f:
    meta = pickle.load(f)

# %%
Cats = enum.IntEnum('Cats', meta.cat_names, start=0)
Conts = enum.IntEnum('Conts', meta.cont_names, start=0)
QCols = enum.IntEnum('QCols', meta.qcols, start=0)
LCols = enum.IntEnum('LCols', meta.lcols, start=0)

# %%
class TutorNet(nn.Module):
    def __init__(self, emb_szs, tag_emb_szs, emb_do, n_cont, trf_dim, trf_enc, trf_dec, trf_heads, trf_do, trf_act):
        super().__init__()
        self.nhead,self.trf_dim = trf_heads, trf_dim
        
        tag_emb_szs =(tag_emb_szs[0]+1, trf_dim)

        self.embeds    = nn.ModuleList([nn.Sequential(nn.Embedding(ni+1, nf, max_norm=1.),nn.Linear(nf,trf_dim)) 
                                        for ni,nf in emb_szs])
        self.tagembeds = nn.EmbeddingBag(*tag_emb_szs, max_norm=1., mode='sum')
            
        self.conts     = nn.Linear(n_cont,trf_dim)
            
        self.trafo = nn.Transformer(
            d_model = trf_dim,
            nhead = trf_heads,
            num_encoder_layers = trf_enc,
            num_decoder_layers = trf_dec,
            dim_feedforward = trf_dim*4,
            dropout = trf_do,
            activation = trf_act,
        )

        self.mlp = nn.Linear(trf_dim, 6)
        
    def forward(self, x_mask, x_cat, x_cont, x_tags, x_tagw, shuffle=None, lam=None):
        b, sl, catf, contf, tagsf = (*x_cat.shape, x_cont.shape[2], x_tags.shape[2])
        
        x_cat  += 1
        x_tags += 1
    
        # compute masks
        causal_mask  = ~torch.tril(torch.ones(1,sl, sl,dtype=torch.bool,device=x_cat.device)).expand(b,-1,-1)
        x_tci   = x_cat[...,Cats.task_container_id]
        x_tci_s = torch.zeros_like(x_tci)
        x_tci_s[...,1:] = x_tci[...,:-1]
        enc_container_aware_mask =  (x_tci.unsqueeze(-1) == x_tci_s.unsqueeze(-1).permute(0,2,1)) | causal_mask
        dec_container_aware_mask = ~(x_tci.unsqueeze(-1) == x_tci.unsqueeze(-1).permute(0,2,1))   & causal_mask

        padding_mask = x_mask 
                
        # encoder x (shifted q & a)
        enc_cat  = torch.zeros_like(x_cat)
        enc_cont = torch.zeros_like(x_cont)
        enc_tags = torch.zeros_like(x_tags)
        enc_tagw = torch.zeros_like(x_tagw)
        
        enc_cat[:,1:]  = x_cat[:,:-1]
        enc_cont[:,1:] = x_cont[:,:-1]
        enc_tags[:,1:] = x_tags[:,:-1]
        enc_tagw[:,1:] = x_tagw[:,:-1]
        
        # decoder x (nonshifted q)
        dec_cat  = x_cat
        dec_cont = x_cont
        dec_tags = x_tags
        dec_tagw = x_tagw

        # hide correct answer and user answered correctly from decoder
        dec_cat[...,Cats.answered_correctly] = 0
        dec_cat[...,Cats.user_answer] = 0
        dec_cat[...,Cats.qhe] = 0
        dec_cont[...,Conts.qet] = 0
        dec_cont[...,Conts.qet_log] = 0
        
        # print(enc_cont.shape)
        enc_cat  =  enc_cat.view(b * sl, catf)   # b*sl, catf
        enc_tags = enc_tags.view(b * sl, tagsf) # b*sl, tagsf
        enc_tagw = enc_tagw.view(b * sl, tagsf) # b*sl, tagsf

        dec_cat  =  dec_cat.view(b * sl, catf)   # b*sl, catf
        dec_tags = dec_tags.view(b * sl, tagsf) # b*sl, tagsf
        dec_tagw = dec_tagw.view(b * sl, tagsf) # b*sl, tagsf
        
        # embed categorical vars
        enc = torch.mean(torch.stack([
            *[ e(enc_cat[:,i]) for i, e in enumerate(self.embeds) ],
            self.tagembeds(enc_tags, per_sample_weights=enc_tagw),
            self.conts(enc_cont).view(-1,self.trf_dim)
        ]),dim=0)
        
        dec = torch.mean(torch.stack([
            *[ e(dec_cat[:,i]) for i, e in enumerate(self.embeds) ],
            self.tagembeds(dec_tags, per_sample_weights=dec_tagw),
            self.conts(dec_cont).view(-1,self.trf_dim)
        ]),dim=0)
        
        enc = enc.view(b, sl, self.trf_dim)           # b, sl, sum of cat, cont and tag ftrs
        dec = dec.view(b, sl, self.trf_dim)           # b, sl, sum of cat, cont and tag ftrs

        if shuffle is not None:
            enc = torch.lerp(enc, enc[shuffle], lam.view(lam.shape[0], 1, 1))
            dec = torch.lerp(dec, dec[shuffle], lam.view(lam.shape[0], 1, 1))
            padding_mask = None
            container_aware_mask |= container_aware_mask[shuffle]
        
        enc = enc.permute(1, 0, 2)          # sl, b, tf (torchformer input)
        dec = dec.permute(1, 0, 2)          # sl, b, tf

        expand_nheads = lambda t: t.unsqueeze(1).expand(t.shape[0],self.nhead,-1,-1).reshape(-1,*t.shape[-2:])
        
        o = self.trafo(
            enc, 
            dec, 
            src_mask = expand_nheads(enc_container_aware_mask),
            tgt_mask = expand_nheads(dec_container_aware_mask),
            memory_mask = expand_nheads(enc_container_aware_mask),
            src_key_padding_mask = padding_mask,
            tgt_key_padding_mask = padding_mask,
            memory_key_padding_mask = padding_mask,
        )                                   # sl, b, tf
        o = o.permute(1, 0, 2)              # b, sl, tf
        o = self.mlp(o)                     # b, sl, of (of=2)
        #print(o)
        return o


# %%
emb_szs = list(zip(meta.n_emb.values(), meta.emb_dim.values()))
tag_emb_szs = meta.tags_n_emb, meta.tags_emb_dim

# %%
class ModelPatcher:
    def __init__(self):
        self.patterns = []

    def is_patchable(self, module_name, module, raiseError):
        return True

    def get_patchable_layers(self, model):
        # Layer names (displayed as regexps)")
        ret = []
        for k, v in model.named_modules():
            if self.is_patchable(k, v, raiseError=False):
                r = re.escape(k)
                ret.append({"regexp": r, "layer": v})
        return ret

    def add_pattern(self, pattern, patch_info):
        self.patterns.append(dict(pattern=pattern, patch_info=patch_info))

    def pattern_match(self, module_name):
        for pattern_def in self.patterns:
            if re.match(pattern_def["pattern"], module_name):
                return True, pattern_def["patch_info"]
        return False, -1

    def new_child_module(self, child_module_name, child_module, patch_info):
        raise NotImplementedError("Implement this in subclasses")

    def replace_module(self, father, child_module_name, child_name, child_module, patch_info):
        new_child_module = self.new_child_module(child_module_name, child_module, patch_info)
        if new_child_module is not None:
            setattr(father, child_name, new_child_module)

    def patch_model(self, model):
        modules = {}
        modified = False
        for k, v in model.named_modules():
            modules[k] = v
            match, patch_info = self.pattern_match(k)
            if match and self.is_patchable(k, v, raiseError=True):
                parts = k.split(".")
                father_module_name = ".".join(parts[:-1])
                child_name = parts[-1]
                father = modules[father_module_name]
                self.replace_module(father, k, child_name, v, patch_info)
                modified = True
        if not modified:
            print(
                "Warning: the patcher did not patch anything!"
                " Check patchable layers with `mp.get_patchable_layers(model)`"
            )


# %%
model1 = TutorNet(emb_szs, tag_emb_szs, H1.emb_do, len(meta.cont_names), H1.trf_dim, H1.trf_enc, H1.trf_dec, H1.trf_heads, H1.trf_do, H1.trf_act)
model2 = TutorNet(emb_szs, tag_emb_szs, H2.emb_do, len(meta.cont_names), H2.trf_dim, H2.trf_enc, H2.trf_dec, H2.trf_heads, H2.trf_do, H2.trf_act)

# %%
def trunc_normal_(x, mean=0., std=1.):
    "Truncated normal initialization (approximation)"
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)

class MyModelPatcher(ModelPatcher):
    def new_child_module(self, child_module_name, child_module, patch_info): return nn.Identity()
mp = MyModelPatcher()
mp.add_pattern(r".*norm\d?.*",{})
    
if H1.tfixup: mp.patch_model(model1)
if H2.tfixup: mp.patch_model(model2)

# %%
state_dict1 = torch.load(in_d / f'{H1.load}', map_location=DEVICE)
state_dict2 = torch.load(in_d / f'{H2.load}', map_location=DEVICE)
# %%
if 'model' in state_dict1:
    state_dict1 = state_dict1['model']
if 'model' in state_dict2:
    state_dict2 = state_dict2['model']

# %%
model1 = model1.to(DEVICE)
model2 = model2.to(DEVICE)

# %%
model1.load_state_dict(state_dict1)
model2.load_state_dict(state_dict2)

# %%
"""
# Infer
"""

# %%
class MyRiiidEnv:
    def __init__(self, p):
        test_dtypes = {
            'group_num': 'int64',
            'row_id': 'int64',
            'timestamp': 'int64',
            'user_id': 'int32',
            'content_id': 'int16',
            'content_type_id': 'int8',
            'task_container_id': 'int16',
            'prior_question_elapsed_time': 'float32',
            'prior_question_had_explanation': 'boolean',
            'prior_group_answers_correct': 'object',
            'prior_group_responses': 'object',
        }
    
        pred_dtypes = {
            'group_num': 'int64',
            'row_id': 'int64',
            'answered_correctly': 'float64',
        }

        self.test_df = pd.read_csv(
            p / f'validation_x_{H.valid_pct}.csv',
            usecols=test_dtypes.keys(),
            dtype=test_dtypes,
        ).set_index('group_num')
        
        self.pred_df = pd.read_csv(
            p / f'validation_submission_{H.valid_pct}.csv',
            usecols=pred_dtypes.keys(),
            dtype=pred_dtypes,
        ).set_index('group_num')
        
        self.first = True

    def iter_test(self):
        for (_, t), (_, p) in zip(self.test_df.groupby('group_num'), self.pred_df.groupby('group_num')):
            yield t, p

    def predict(self, p):
        if self.first:
            p.to_csv('submission.csv', index=False)
            self.first = False
        else:
            p.to_csv('submission.csv', index=False, mode='a', header=False)

def make_env():
    return MyRiiidEnv(Path('input'))


# %%
if KAGGLE:
    import riiideducation
    env = riiideducation.make_env()
else:
    env = make_env()

# %%
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

# %%
def get_x(user_ids, cat_names, cont_names, hist_cat_d, hist_cont_d, hist_tags_d, hist_tagw_d, chunk_size):
    num_rows_per_uid_d = defaultdict(int)
    for uid in user_ids:
        num_rows_per_uid_d[uid] += 1

    n_users = len(num_rows_per_uid_d)

    # prepare the batch
    x_mask = np.ones ((n_users, chunk_size), dtype=np.bool)
    x_cat  = np.zeros((n_users, chunk_size, len(cat_names)),  dtype=np.long)
    x_cont = np.full ((n_users, chunk_size, len(cont_names)), np.nan, dtype=np.float32)
    x_tags = np.zeros((n_users, chunk_size, 6), dtype=np.long)
    x_tagw = np.zeros((n_users, chunk_size, 6), dtype=np.float32)
    
    for i, uid in enumerate(num_rows_per_uid_d.keys()):
        # trim history
        hist_cat_d [uid] = hist_cat_d [uid][-chunk_size:]
        hist_cont_d[uid] = hist_cont_d[uid][-chunk_size:]
        hist_tags_d[uid] = hist_tags_d[uid][-chunk_size:]
        hist_tagw_d[uid] = hist_tagw_d[uid][-chunk_size:]

        sl = hist_cat_d[uid].shape[0]
        
        x_mask[i,:sl] = False
        x_cat [i,:sl] = hist_cat_d[uid]
        x_cont[i,:sl] = hist_cont_d[uid]
        x_tags[i,:sl] = hist_tags_d[uid]
        x_tagw[i,:sl] = hist_tagw_d[uid]
    
    return x_mask, x_cat, x_cont, x_tags, x_tagw

# %%
def get_preds(user_ids, preds, x_mask):
    poi = torch.zeros(len(user_ids), 2, device=preds.device) # predictions of interest

    user_row = defaultdict(lambda: len(user_row))
    for uid in user_ids:
        user_row[uid]

    poi_idxs = { uid: torch.from_numpy(user_ids == uid)  for uid in user_row.keys() }
    
    for uid in user_ids:
        ur = user_row[uid]           # user row (1st dim) of the preds tensor
        pi = poi_idxs[uid]           # indexes to the original locations of the interactions
        x = preds[ur,~x_mask[ur],:2] # get all predictions (both history and new)
        x = x[-pi.sum():]            # the last pi.sum() preds are the new ones
        poi[pi] = x
        
    return torch.softmax(poi, dim=-1)[:,1].detach().cpu()

# %%
from numba import njit
from scipy.stats import rankdata

@njit
def _auc(actual, pred_ranks):
    actual = np.asarray(actual)
    pred_ranks = np.asarray(pred_ranks)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(pred_ranks[actual==1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

def auc(actual, predicted):
    pred_ranks = rankdata(predicted)
    return _auc(actual, pred_ranks)

# %%
def my_roc_auc(pred, targ):
    idx = targ != -1
    pred = pred[idx]
    targ = targ[idx]
    return auc(targ.cpu().numpy(), pred.cpu().numpy())

# %%
previous_users = None
means = torch.from_numpy(meta.means).to(DEVICE)
stds  = torch.from_numpy(meta.stds).to(DEVICE)

# %%
if not KAGGLE:
    # delete history of validation users
    val_users = users_list[int((1 - H.valid_pct) * len(users_list)):]
    for val_user in val_users:
        encoded_val_user = users_d[val_user]
        if val_user in data.cat_d:
            del data.cat_d[val_user]
            del data.cont_d[val_user]
            del data.tags_d[val_user]
            del data.tagw_d[val_user]
            del data.last_q_container_id_d[val_user]
            data.last_ts[encoded_val_user,0] = 0
            attempt_num[encoded_val_user] = 0
            attempts_correct[encoded_val_user] = 0

# %%
model1 = model1.eval()
model2 = model2.eval()

# %%
n_read_rows = 0
n_predicted_rows = 0
n_predicted_rows_by_model_2 = 0
inference_start_time = time.time()
flag_ensemble = True

Col = None
prior_user_ids = None # linter go away
all_preds = torch.FloatTensor()
all_targs = torch.LongTensor()

pbar = tqdm(env.iter_test())

inference_count = 0
for test_df, pred_df in pbar:
    if inference_count > 90:
        break
    if Col is None:
        Col = enum.IntEnum('Col', test_df.columns.tolist(), start=0)

    prior_group_answers_correct = np.fromstring(test_df.iloc[0].prior_group_answers_correct[1:-1], dtype=np.int16, sep=',')
    prior_group_responses       = np.fromstring(test_df.iloc[0].prior_group_responses      [1:-1], dtype=np.int16, sep=',')

    if MODE == Mode.hurry_up and n_read_rows > HURRY_UP_CUTOFF * 2.5e6:
        preds = torch.full((len(test_df),), 0.5)
    else:
        if prior_group_responses.size > 0: update_answers(
            prior_user_ids,
            prior_group_answers_correct,
            prior_group_responses,
            meta.cat_names,
            meta.cont_names,
            meta.codes_d,
            data.cat_d,
            data.cont_d,
            users_d,
            attempt_num,
            attempts_correct
        )

        prior_user_ids = update_questions(
            test_df, 
            Col,
            meta.cat_names, 
            meta.cont_names, 
            meta.qc_d, 
            meta.lc_d,
            meta.codes_d, 
            QCols, 
            LCols, 
            Cats,
            Conts, 
            data.cat_d, 
            data.cont_d, 
            data.tags_d, 
            data.tagw_d, 
            data.last_q_container_id_d,
            data.last_ts, 
            attempt_num,
            attempts_correct,
            data.qp_d,
            users_d
        )
            
        # get x
        a_mask, a_cat, a_cont, a_tags, a_tagw = get_x(
            prior_user_ids,
            meta.cat_names,
            meta.cont_names,
            data.cat_d,
            data.cont_d,
            data.tags_d,
            data.tagw_d,
            H.chunk_size
        )

        # Predict
        if MODE == Mode.blindfolded_gunslinger and n_read_rows < BLIND_CUTOFF * 2.5e6:
            preds = torch.full((len(test_df),), 0.5)
            inference_start_time = time.time()
        else:
            with torch.no_grad():
                batch_preds1 = torch.FloatTensor().to(DEVICE)
                if flag_ensemble:
                    batch_preds2 = torch.FloatTensor().to(DEVICE)
                for b in range((a_cat.shape[0] + H.bs - 1) // H.bs):
                    x_mask = torch.from_numpy(a_mask[b*H.bs:(b+1)*H.bs]).to(DEVICE)
                    x_cat  = torch.from_numpy(a_cat [b*H.bs:(b+1)*H.bs]).to(DEVICE)
                    x_cont = torch.from_numpy(a_cont[b*H.bs:(b+1)*H.bs]).to(DEVICE)
                    x_tags = torch.from_numpy(a_tags[b*H.bs:(b+1)*H.bs]).to(DEVICE)
                    x_tagw = torch.from_numpy(a_tagw[b*H.bs:(b+1)*H.bs]).to(DEVICE)

                    # Normalize x_cont on GPU and take care of nans
                    x_cont = (x_cont - means) / stds
                    x_cont[torch.isnan(x_cont)] = 0.
                    x_cont = x_cont.to(torch.float32)

                    pred_time_start_1 = time.time()
                    if flag_ensemble:
                        preds1 = model1(x_mask.clone(), x_cat.clone(), x_cont.clone(), x_tags.clone(), x_tagw.clone())
                    else:
                        preds1 = model1(x_mask, x_cat, x_cont, x_tags, x_tagw)
                    pred_time_end_1 = time.time()
                    batch_preds1 = torch.cat([batch_preds1, preds1])

                    if flag_ensemble:
                        pred_time_start_2 = time.time()
                        preds2 = model2(x_mask, x_cat, x_cont, x_tags, x_tagw)
                        pred_time_end_2 = time.time()
                        batch_preds2 = torch.cat([batch_preds2, preds2])

                preds = get_preds(prior_user_ids, batch_preds1, torch.from_numpy(a_mask).to(DEVICE))

                n_predicted_rows += len(test_df)
                if flag_ensemble:
                    preds2 = get_preds(prior_user_ids, batch_preds2, torch.from_numpy(a_mask).to(DEVICE))
                    preds = (preds + preds2) / 2
                    n_predicted_rows_by_model_2 += len(test_df)

    test_df['answered_correctly'] = preds
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

    # adaptive ensembling
    n_read_rows += len(test_df)
    if n_predicted_rows < 1000:
        flag_ensemble = True
    else:
        elapsed_inference_time = time.time() - inference_start_time
        estimated_total_inference_time = elapsed_inference_time * ROWS_TO_INFER / n_predicted_rows
        startup_time = inference_start_time - start_time
        flag_ensemble = estimated_total_inference_time < (TIME_BUDGET - startup_time)

    if not KAGGLE:
        all_preds = torch.cat([all_preds, preds])
        all_targs = torch.cat([all_targs, torch.LongTensor(prior_group_answers_correct)])
        pub_preds = all_preds[:int(PUB_PVT_CUTOFF * 2.5e6)]
        pub_targs = all_targs[:int(PUB_PVT_CUTOFF * 2.5e6)]
        pvt_preds = all_preds[int(PUB_PVT_CUTOFF * 2.5e6):]
        pvt_targs = all_targs[int(PUB_PVT_CUTOFF * 2.5e6):]
        postfix = {
            'model 1 pred rows': n_predicted_rows,
            'model 2 pred rows': n_predicted_rows_by_model_2,
        }
        postfix['model1 pred time(sec)'] = f'{(pred_time_end_1 - pred_time_start_1):.3f}'
        postfix['model2 pred time(sec)'] = f'{(pred_time_end_2 - pred_time_start_2):.3f}'
        if n_predicted_rows >= 1000:
            postfix['eta'] = f'{estimated_total_inference_time / 60 / 60:.3f}/{(TIME_BUDGET - startup_time) / 60 / 60:.3f}'

        if len(pub_targs) > 0:
            pub_auroc = my_roc_auc(pub_preds[:len(pub_targs)], pub_targs)
            postfix['auroc (pub)'] = f'{pub_auroc:.6f}'
        if len(pvt_targs) > 0:
            pvt_auroc = my_roc_auc(pvt_preds[:len(pvt_targs)], pvt_targs)
            postfix['auroc (pvt)'] = f'{pvt_auroc:.6f}'
        pbar.set_postfix(postfix)
    
    inference_count += 1



# %%
if KAGGLE:
    shutil.rmtree('resources')

elapsed_time = time.time() - start_time

print(f'Elapsed time: {elapsed_time:5.2f}s')
