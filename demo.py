#!python
import tqdm
import cherrypy
import cherrypy_cors
import argparse
import time
import datetime
from datetime import datetime 
import random
from matres_reader_with_tense import *
import os
from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader
from util import *
from pprint import pprint
from transformers import AutoTokenizer, AutoModel
from model import transformers_mlp_cons
from exp import *
import numpy as np
import json
import sys
from synonyms import *
import pickle
from timeline_construct import *
from ts import func, ModelWithTemperature
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# datetime object containing current date and time
now = datetime.datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
print("date and time =", dt_string)

#label_dict={"SuperSub": 0, "SubSuper": 1, "Coref": 2, "NoRel": 3}
num_dict = {0: "before", 1: "after", 2: "equal", 3: "vague"}
#def label_to_num(label):
#    return label_dict[label]
def num_to_label(num):
    return num_dict[num]

mask_in_input_ids = 0 # note that [MASK] is actually learned through pre-training
mask_in_input_mask = 0 # when input is masked through attention, it would be replaced with [PAD]
acronym = 0 # using acronym for tense (e.g., pastsimp): 1; else (e.g., past simple): 0
t_marker = 1
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
        
def docTransformerTokenIDs(sentences):
    if len(sentences) < 1:
        return None
    elif len(sentences) == 1:
        return sentences[0]['_subword_to_ID']
    else:
        TokenIDs = sentences[0]['_subword_to_ID']
        for i in range(1, len(sentences)):
            TokenIDs += sentences[i]['_subword_to_ID'][1:]
        return TokenIDs
    
def collate_fn(batch):
    max_len = max([len(f['input_ids']) for f in batch])
    input_ids = [f['input_ids'] + [0] * (max_len - len(f['input_ids'])) for f in batch]
    if mask_in_input_ids:
        input_ids_new = []
        for f_id, f in enumerate(input_ids):
            for event_id, start in enumerate(batch[f_id]['event_pos']):
                end = batch[f_id]['event_pos_end'][event_id]
                for token_id in range(start, end): # needs verification
                    f[token_id] = 67
            input_ids_new.append(f)
        input_ids = input_ids_new
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = [[1.0] * len(f['input_ids']) + [0.0] * (max_len - len(f['input_ids'])) for f in batch]
    if mask_in_input_mask:
        input_mask_new = []
        for f_id, f in enumerate(input_mask):
            for event_id, start in enumerate(batch[f_id]['event_pos']):
                end = batch[f_id]['event_pos_end'][event_id]
                for token_id in range(start, end): # needs verification
                    f[token_id] = 0.0
            input_mask_new.append(f)
        input_mask = input_mask_new
    # Updated on May 17, 2022    
    input_mask_eo = [[0.0] * max_len for f in batch]
    for f_id, f in enumerate(input_mask_eo):
        for event_id, start in enumerate(batch[f_id]['event_pos']):
            end = batch[f_id]['event_pos_end'][event_id]
            for token_id in range(start, end): # needs verification
                f[token_id] = 1.0
    # Updated on Jun 14, 2022
    input_mask_xbar = [[0.0] * max_len for f in batch]
    input_mask_xbar = torch.tensor(input_mask_xbar, dtype=torch.float)
    input_mask_eo = torch.tensor(input_mask_eo, dtype=torch.float)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    event_pos = [f['event_pos'] for f in batch]
    event_pos_end = [f['event_pos_end'] for f in batch]
    event_pair = [f['event_pair'] for f in batch]
    labels = [f['labels'] for f in batch]
    output = (input_ids, input_mask, event_pos, event_pos_end, event_pair, labels, input_mask_eo, input_mask_xbar)
    return output

#############################
### Setting up parameters ###
#############################
f1_metric = 'micro'
params = {'transformers_model': 'google/bigbird-roberta-large',
          'dataset': 'MATRES',   # 'HiEve', 'IC', 'MATRES' 
          'testdata': 'PRED', # MATRES / MATRES_nd / TDD / PRED / None; None means training mode
          'block_size': 64,
          'add_loss': 0, 
          'batch_size': 1,    # 6 works on 48G gpu
          'epochs': 40,
          'learning_rate': 5e-6,    # subject to change
          'seed': 0,
          'gpu_id': '11453',    # subject to change
          'debug': 0,
          'rst_file_name': '0511pm-lr5e-6-b20-gpu9942-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td1-dpn1-mask0.rst',    # subject to change
          'mask_in_input_ids': mask_in_input_ids,
          'mask_in_input_mask': mask_in_input_mask,
          'marker': 'abc', 
          'tense_acron': 0, # 1 (acronym of tense) or 0 (original tense)
          't_marker': 1, # 2 (trigger enclosed by special tokens) or 1 (tense enclosed by **)
          'td': 1, # 0 (no tense detection) or 1 (tense detection, add tense info)
          'dpn': 1, # 1 if use DPN; else 0
          'lambda_1': -10, # lower bound * 10
          'lambda_2': 11, # upper bound * 10
          'f1_metric': f1_metric, 
         }
# $acr $tmarker $td $dpn $mask $lambda_1 $lambda_2

# FOR 48GBgpu
if params['testdata'] in ['MATRES', 'MATRES_nd']:
    #params['batch_size'] = 400
    params['batch_size'] = 1
if params['testdata'] in ['TDD']:
    params['batch_size'] = 100
    
if params['testdata'] == 'MATRES_nd':
    params['nd'] = True
else:
    params['nd'] = False
    
###########
# NO MASK #
###########
if params['rst_file_name'] == '0414am-lr5e-6-b20-gpu2-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td0-dpn1.rst':
    slurm_id = '10060'
#params['rst_file_name'] = '0414am-lr5e-6-b20-gpu2-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td0-dpn1.rst' 
#slurm_id = '10060'
# python main_pair.py 0615_10060.rst 5e-6 400 10060 0 MATRES abc 0 1 0 1 0 -10 11
# python main_pair.py 0414am-lr5e-6-b20-gpu2-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td0-dpn1.rst 5e-6 400 10060 0 MATRES abc 0 1 0 1 0 -10 11

if params['rst_file_name'] == '0419pm-lr5e-6-b20-gpu4-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td0-dpn0.rst':
    slurm_id = '10489'
#params['rst_file_name'] = '0419pm-lr5e-6-b20-gpu4-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td0-dpn0.rst' 
#slurm_id = '10489'
# python main_pair.py 0615_10489.rst 5e-6 400 10489 0 MATRES abc 0 1 0 0 0 -10 11
# python main_pair.py 0419pm-lr5e-6-b20-gpu4-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td0-dpn0.rst 5e-6 400 10489 0 MATRES abc 0 1 0 0 0 -10 11

if params['rst_file_name'] == '0511pm-lr5e-6-b20-gpu9942-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td1-dpn1-mask0.rst':
    slurm_id = '11453'
#params['rst_file_name'] = '0511pm-lr5e-6-b20-gpu9942-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td1-dpn1-mask0.rst' 
#slurm_id = '11453'
# python main_pair.py 0615_11453.rst 5e-6 400 11453 0 MATRES abc 0 1 1 1 0 -10 11
# python main_pair.py 0511pm-lr5e-6-b20-gpu9942-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td1-dpn1-mask0.rst 5e-6 400 11453 0 MATRES abc 0 1 1 1 0 -10 11

if params['rst_file_name'] == '0419pm-lr5e-6-b20-gpu5-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td1-dpn0.rst':
    slurm_id = '10488'
#params['rst_file_name'] = '0419pm-lr5e-6-b20-gpu5-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td1-dpn0.rst'
#slurm_id = '10488'
# python main_pair.py 0615_10488.rst 5e-6 400 10488 0 MATRES abc 0 1 1 0 0 -10 11
# python main_pair.py 0419pm-lr5e-6-b20-gpu5-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td1-dpn0.rst 5e-6 400 10488 0 MATRES abc 0 1 1 0 0 -10 11

########
# MASK #
########

if params['rst_file_name'] == '0414am-lr5e-6-b20-gpu3-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td0-dpn1.rst':
    slurm_id = '10063'
#params['rst_file_name'] = '0414am-lr5e-6-b20-gpu3-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td0-dpn1.rst'
#slurm_id = '10063'
# python main_pair.py 0614_10063.rst 5e-6 400 10063 0 MATRES abc 0 1 0 1 1 -10 11
# python main_pair.py 0414am-lr5e-6-b20-gpu3-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td0-dpn1.rst 5e-6 400 10063 0 MATRES abc 0 1 0 1 1 -10 11

if params['rst_file_name'] == '0419pm-lr5e-6-b20-gpu3-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td0-dpn0.rst':
    slurm_id = '10490'
#params['rst_file_name'] = '0419pm-lr5e-6-b20-gpu3-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td0-dpn0.rst'
#slurm_id = '10490'
# python main_pair.py 0614_10490.rst 5e-6 400 10490 0 MATRES abc 0 1 0 0 1 -10 11
# python main_pair.py 0419pm-lr5e-6-b20-gpu3-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td0-dpn0.rst 5e-6 400 10490 0 MATRES abc 0 1 0 0 1 -10 11

if params['rst_file_name'] == '0511pm-lr5e-6-b20-gpu9937-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td1-dpn1-mask1.rst':
    slurm_id = '11454'
#params['rst_file_name'] = '0511pm-lr5e-6-b20-gpu9937-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td1-dpn1-mask1.rst' 
#slurm_id = '11454'
# python main_pair.py 0614_11454.rst 5e-6 400 11454 0 MATRES abc 0 1 1 1 1 -10 11
# python main_pair.py 0511pm-lr5e-6-b20-gpu9937-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td1-dpn1-mask1.rst 5e-6 400 11454 0 MATRES abc 0 1 1 1 1 -10 11

if params['rst_file_name'] == '0419pm-lr5e-6-b20-gpu6-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td1-dpn0.rst':
    slurm_id = '10487'
#params['rst_file_name'] = '0419pm-lr5e-6-b20-gpu6-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td1-dpn0.rst'
#slurm_id = '10487'
# python main_pair.py 0614_10487.rst 5e-6 400 10487 0 MATRES abc 0 1 1 0 1 -10 11 
# python main_pair.py 0419pm-lr5e-6-b20-gpu6-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td1-dpn0.rst 5e-6 400 10487 0 MATRES abc 0 1 1 0 1 -10 11 

if params['transformers_model'][-5:] == "large":
    params['emb_size'] = 1024
elif params['transformers_model'][-4:] == "base":
    params['emb_size'] = 768
else:
    print("emb_size is neither 1024 nor 768? ...")
    
set_seed(params['seed'])
rst_file_name = params['rst_file_name']
"""
model_params_dir = "./model_params/"
if params['dataset'] == 'HiEve':
    best_PATH = model_params_dir + "HiEve_best/" + rst_file_name.replace(".rst", ".pt") # to save model params here
elif params['dataset'] == 'IC':
    best_PATH = model_params_dir + "IC_best/" + rst_file_name.replace(".rst", ".pt") # to save model params here
elif params['dataset'] == 'MATRES':
    best_PATH = model_params_dir + "MATRES_best/" + rst_file_name.replace(".rst", ".pt") # to save model params here
else:
    print("Dataset unknown...")
"""
best_PATH = sys.argv[1] + '/' + '0511.pt' 
model_name = rst_file_name.replace(".rst", "")
with open("config/" + rst_file_name.replace("rst", "json"), 'w') as config_file:
    json.dump(params, config_file)
    
if int(params['gpu_id']) < 10:
    os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu_id']
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
cuda = torch.device('cuda')
params['cuda'] = cuda # not included in config file

#######################
### Data processing ###
#######################

print("Processing " + params['dataset'] + " dataset...")
t0 = time.time()
if params['dataset'] == "IC":
    dir_name = "./IC/IC_Processed/"
    #max_sent_len = 193
elif params['dataset'] == "HiEve":
    dir_name = "./hievents_v2/processed/"
    #max_sent_len = 155
elif params['dataset'] == "MATRES":
    dir_name = ""
else:
    print("Not supporting this dataset yet!")
    
tokenizer = AutoTokenizer.from_pretrained(params['transformers_model'])   
if acronym:
    special_tokens_dict = {'additional_special_tokens': 
                           [' [futuperfsimp]',' [futucont]',' [futuperfcont]',' [futusimp]', ' [pastcont]', ' [pastperfcont]', ' [pastperfsimp]', ' [pastsimp]', ' [prescont]', ' [presperfcont]', ' [presperfsimp]', ' [pressimp]', ' [futuperfsimppass]',' [futucontpass]',' [futuperfcontpass]',' [futusimppass]', ' [pastcontpass]', ' [pastperfcontpass]', ' [pastperfsimppass]', ' [pastsimppass]', ' [prescontpass]', ' [presperfcontpass]', ' [presperfsimppass]', ' [pressimppass]', ' [none]'
                           ]}
    spec_toke_list = []
    for t in special_tokens_dict['additional_special_tokens']:
        spec_toke_list.append(" [/" + t[2:])
    special_tokens_dict['additional_special_tokens'] += spec_toke_list
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model = AutoModel.from_pretrained(params['transformers_model'])
    model.resize_token_embeddings(len(tokenizer))
else:
    model = AutoModel.from_pretrained(params['transformers_model'])
params['model'] = model
debug = params['debug']
if debug:
    params['epochs'] = 1

def add_tense_info(x_sent, tense, start, mention, special_1, special_2):
    # x:
    # special_1: 2589
    # special_2: 1736
    
    # y:
    # special_1: 1404
    # special_2: 5400
    orig_len = len(x_sent)        
    if tense:
        if acronym:
            tense_marker = tokenizer.encode(" " + tense[acronym])[1:-1]
        else:
            tense_marker = tokenizer.encode(tense[acronym])[1:-1]
    else:
        if acronym:
            tense_marker = tokenizer.encode(" [none]")[1:-1]
        else:
            tense_marker = tokenizer.encode("None")[1:-1]
    subword_len = len(tokenizer.encode(mention)) - 2
    if t_marker == 2:
        # trigger enclosed by special tense tokens
        assert acronym == 1
        x_sent = x_sent[0:start] + tense_marker + x_sent[start:start+subword_len] + tokenizer.encode(" [/" + tokenizer.decode(tense_marker)[2:])[1:-1] + x_sent[start+subword_len:]
        new_start = start + len(tense_marker)
    elif t_marker == 1:
        # tense enclosed by * *
        x_sent = x_sent[0:start] + [special_1, special_2] + tense_marker + [special_2] + x_sent[start:start+subword_len] + [special_1] + x_sent[start+subword_len:]
        new_start = start + len([special_1, special_2] + tense_marker + [special_2])
    new_end = new_start + subword_len
    offset = len(x_sent) - orig_len
    return x_sent, offset, new_start, new_end

def reverse_num(event_position):
    return [event_position[1], event_position[0]]
  
'''
# if running MATRES, the data processing starts here:
##############
### MATRES ###
##############  

doc_id = -1
features_train = []
features_valid = []
features_test = []
t0 = time.time()
relation_stats = {0: 0, 1: 0, 2: 0, 3: 0}
t_marker = params['t_marker']
# 2: will [futusimp] begin [/futusimp]
# 1: will @ * Future Simple * begin @ 

max_len = 0
sent_num = 0
pair_num = 0
test_labels = []
context_len = {}
timeline_input = []
for fname in tqdm.tqdm(eiid_pair_to_label.keys()):
    file_name = fname + ".tml"
    if file_name in onlyfiles_TB:
        dir_name = mypath_TB
    elif file_name in onlyfiles_AQ:
        dir_name = mypath_AQ
    elif file_name in onlyfiles_PL:
        dir_name = mypath_PL
    else:
        continue
    my_dict = tml_reader(dir_name, file_name, tokenizer) 
    
    for (eiid1, eiid2) in eiid_pair_to_label[fname].keys():
        pair_num += 1
        event_pos = []
        event_pos_end = []
        relations = []
        TokenIDs = [65]
        x = my_dict["eiid_dict"][eiid1]["eID"] # eID
        y = my_dict["eiid_dict"][eiid2]["eID"]
        x_sent_id = my_dict["event_dict"][x]["sent_id"]
        y_sent_id = my_dict["event_dict"][y]["sent_id"]
        reverse = False
        if x_sent_id > y_sent_id:
            reverse = True
            x = my_dict["eiid_dict"][eiid2]["eID"]
            y = my_dict["eiid_dict"][eiid1]["eID"]
            x_sent_id = my_dict["event_dict"][x]["sent_id"]
            y_sent_id = my_dict["event_dict"][y]["sent_id"]
        elif x_sent_id == y_sent_id:
            x_position = my_dict["event_dict"][x]["_subword_id"]
            y_position = my_dict["event_dict"][y]["_subword_id"]
            if x_position > y_position:
                reverse = True
                x = my_dict["eiid_dict"][eiid2]["eID"]
                y = my_dict["eiid_dict"][eiid1]["eID"]
        x_sent = my_dict["sentences"][x_sent_id]["_subword_to_ID"]
        y_sent = my_dict["sentences"][y_sent_id]["_subword_to_ID"]
        # This guarantees that trigger x is always before trigger y in narrative order

        context_start_sent_id = max(x_sent_id-1, 0)
        context_end_sent_id = min(y_sent_id+2, len(my_dict["sentences"]))
        c_len = context_end_sent_id - context_start_sent_id
        if c_len in context_len.keys():
            context_len[c_len] += 1
        else:
            context_len[c_len] = 1
        sent_num += c_len
        
        if params['td'] == 1:
            x_sent, offset_x, new_start_x, new_end_x = add_tense_info(x_sent, my_dict["event_dict"][x]['tense'], my_dict['event_dict'][x]['_subword_id'], my_dict["event_dict"][x]['mention'], 2589, 1736)
        else:
            x_sent, offset_x, new_start_x, new_end_x = x_sent, 0, my_dict['event_dict'][x]['_subword_id'], my_dict['event_dict'][x]['_subword_id'] + len(tokenizer.encode(my_dict["event_dict"][x]['mention'])) - 2
            
        if x_sent_id != y_sent_id:
            if params['td'] == 1:
                y_sent, offset_y, new_start_y, new_end_y = add_tense_info(y_sent, my_dict["event_dict"][y]['tense'], my_dict['event_dict'][y]['_subword_id'], my_dict["event_dict"][y]['mention'], 1404, 5400)
            else:
                y_sent, offset_y, new_start_y, new_end_y = y_sent, 0, my_dict['event_dict'][y]['_subword_id'], my_dict['event_dict'][y]['_subword_id'] + len(tokenizer.encode(my_dict["event_dict"][y]['mention'])) - 2
            for sid in range(context_start_sent_id, context_end_sent_id):
                if sid == x_sent_id:
                    event_pos.append(new_start_x + len(TokenIDs) - 1)
                    event_pos_end.append(new_end_x + len(TokenIDs) - 1)
                    TokenIDs += x_sent[1:]
                elif sid == y_sent_id:
                    event_pos.append(new_start_y + len(TokenIDs) - 1)
                    event_pos_end.append(new_end_y + len(TokenIDs) - 1)
                    TokenIDs += y_sent[1:]
                else:
                    TokenIDs += my_dict["sentences"][sid]["_subword_to_ID"][1:]
        else:
            if params['td'] == 1:
                y_sent, offset_y, new_start_y, new_end_y = add_tense_info(x_sent, my_dict["event_dict"][y]['tense'], my_dict['event_dict'][y]['_subword_id'] + offset_x, my_dict["event_dict"][y]['mention'], 1404, 5400)
            else:
                y_sent, offset_y, new_start_y, new_end_y = y_sent, 0, my_dict['event_dict'][y]['_subword_id'], my_dict['event_dict'][y]['_subword_id'] + len(tokenizer.encode(my_dict["event_dict"][y]['mention'])) - 2
            for sid in range(context_start_sent_id, context_end_sent_id):
                if sid == y_sent_id:
                    event_pos.append(new_start_x + len(TokenIDs) - 1)
                    event_pos_end.append(new_end_x + len(TokenIDs) - 1)
                    event_pos.append(new_start_y + len(TokenIDs) - 1)
                    event_pos_end.append(new_end_y + len(TokenIDs) - 1)
                    TokenIDs += y_sent[1:]
                else:
                    TokenIDs += my_dict["sentences"][sid]["_subword_to_ID"][1:]
                    
        if reverse:
            event_pos = reverse_num(event_pos)
            event_pos_end = reverse_num(event_pos_end)
            
        xy = eiid_pair_to_label[fname][(eiid1, eiid2)]
        
        relations.append(xy)
        relation_stats[xy] += 1
        if len(TokenIDs) > max_len:
            max_len = len(TokenIDs)
        
        if debug or pair_num < 5:
            print("first event of the pair:", tokenizer.decode(TokenIDs[event_pos[0]:event_pos_end[0]]))
            print("second event of the pair:", tokenizer.decode(TokenIDs[event_pos[1]:event_pos_end[1]]))
            print("TokenIDs:", tokenizer.decode(TokenIDs))
        
        if params['nd']:
            syn_0 = replace_with_syn(tokenizer.decode(TokenIDs[event_pos[0]:event_pos_end[0]]))
            syn_1 = replace_with_syn(tokenizer.decode(TokenIDs[event_pos[1]:event_pos_end[1]]))
            if len(syn_0) > 0:
                TokenIDs = TokenIDs[0:event_pos[0]] + tokenizer.encode(syn_0[0])[1:-1] + TokenIDs[event_pos_end[0]:]
                prev = event_pos_end[0]
                event_pos_end[0] = event_pos[0] + len(tokenizer.encode(syn_0[0])[1:-1])
                if prev != event_pos_end[0]:
                    offset = event_pos_end[0] - prev
                    event_pos[1] += offset
                    event_pos_end[1] += offset
            if len(syn_1) > 0:
                TokenIDs = TokenIDs[0:event_pos[1]] + tokenizer.encode(syn_1[0])[1:-1] + TokenIDs[event_pos_end[1]:]
                prev = event_pos_end[1]
                event_pos_end[1] = event_pos[1] + len(tokenizer.encode(syn_1[0])[1:-1])
            #assert 1 == 0
        feature = {'input_ids': TokenIDs,
                   'event_pos': event_pos,
                   'event_pos_end': event_pos_end,
                   'event_pair': [[1, 2]],
                   'labels': relations,
                  }
        if file_name in onlyfiles_TB:
            features_train.append(feature)
        elif file_name in onlyfiles_AQ:
            features_valid.append(feature)
        elif file_name in onlyfiles_PL:
            features_test.append(feature)
            test_labels.append(xy)
            timeline_input.append([fname, x, y, xy])
    if debug:
        break
        
elapsed = format_time(time.time() - t0)
print("MATRES Preprocessing took {:}".format(elapsed)) 
print("Temporal Relation Stats:", relation_stats)
print("Total num of pairs:", pair_num)
print("Max length of context:", max_len)
print("Avg num of sentences that context contains:", sent_num/pair_num)
print("Context length stats(unit: sentence): ", context_len)
print("MATRES train valid test pair num:", len(features_train), len(features_valid), len(features_test))
#with open("MATRES_test_timeline_input.json", 'w') as f:
#    json.dump(timeline_input, f)
#    assert 0 == 1
    
#output_file = open('test_labels.txt', 'w')
#for label in test_labels:
#    output_file.write(str(label) + '\n')
#output_file.close()
#if debug:
#    assert 0 == 1

#### MATRES PROCESSING ENDS HERE ####
'''


"""
################
### HiEve/IC ###
################
for file_name in tqdm.tqdm(onlyfiles):
    doc_id += 1
    my_dict = tsvx_reader(params['dataset'], dir_name, file_name, tokenizer, 0) # 0 if no eventseg
    TokenIDs = docTransformerTokenIDs(my_dict['sentences'])
    event_pos = []
    event_pos_end = []
    for event_id in my_dict['event_dict'].keys():
        sent_id = my_dict['event_dict'][event_id]['sent_id']
        start = my_dict['end_pos'][sent_id] - 1 + my_dict['event_dict'][event_id]['_subword_id'] # 0 x x x x 0 x x x 0
        event_pos.append(start)    
        subword_len = len(tokenizer.encode(my_dict["event_dict"][event_id]['mention'])) - 2
        event_pos_end.append(start + subword_len)
        print(tokenizer.decode([TokenIDs[start]]))
        
    pairs = []
    relations = []
    for rel in my_dict['relation_dict'].keys():
        pairs.append([rel[0], rel[1]])
        relations.append(my_dict['relation_dict'][rel]['relation'])
        
    feature = {'input_ids': TokenIDs,
               'event_pos': event_pos,
               'event_pos_end': event_pos_end,
               'event_pair': pairs,
               'labels': relations,
              }
    features.append(feature)
    
#### HiEve/IC PROCESSING ENDS HERE ####
"""


###################
### TDDiscourse ###
###################
if params['testdata'] == "TDD":
    with open("t5_TDD_dic.pickle", 'rb') as file:
        t5_TDD_dic = pickle.load(file)
        
# TO GENERATE EXAMPLE INPUT FOR PREDICTION
#PRED_FILE = open('example/temporal_example_input.json', 'w')

def TDD_processor(split):
    relation_stats = {0: 0, 1: 0, 2: 0, 3: 0}
    max_len = 0
    sent_num = 0
    pair_num = 0
    features = []
    labels = []
    labels_full = {}
    abnormal_articles = set()
    instance_id = -1
    for art_id in t5_TDD_dic[split].keys():
        tup_id = 0
        for tup in t5_TDD_dic[split][art_id]:
            tup_id += 1
            instance_id += 1
            text, event_pos, event_pos_end, event_ids, e_dict, timeline, abnormal = convert_t5_input(tup[0][19:], tup[1])
            # TO GENERATE EXAMPLE INPUT FOR PREDICTION
            #temp_input = [{'text': text, 'event_pos': event_pos, 'event_pos_end': event_pos_end}]
            #json.dump(temp_input, PRED_FILE)
            #PRED_FILE.close()
            #assert 0 == 1
            if abnormal:
                #print(art_id, tup_id)
                abnormal_articles.add(art_id)
                continue
            labels_full[instance_id] = {"art_id": art_id, "tup_id": tup_id, "timeline": timeline, "event_pairs": []}
            my_dict = tdd_reader(text, event_pos, event_pos_end, tokenizer)
            rev_ind = {}
            for order, eid in enumerate(timeline):
                rev_ind[eid] = order
            for i in range(0, len(timeline)):
                for j in range(i+1, len(timeline)):
                    pair_num += 1
                    event_pos = []
                    event_pos_end = []
                    relations = []
                    TokenIDs = [65]
                    x, y = i, j
                    x_sent_id = my_dict["event_dict"][i]["sent_id"]
                    y_sent_id = my_dict["event_dict"][j]["sent_id"]
                    if x_sent_id > y_sent_id:
                        x, y = j, i
                        x_sent_id = my_dict["event_dict"][x]["sent_id"]
                        y_sent_id = my_dict["event_dict"][y]["sent_id"]
                    elif x_sent_id == y_sent_id:
                        x_position = my_dict["event_dict"][x]["_subword_id"]
                        y_position = my_dict["event_dict"][y]["_subword_id"]
                        if x_position > y_position:
                            x, y = j, i
                    else:
                        x, y = i, j
                    x_sent = my_dict["sentences"][x_sent_id]["_subword_to_ID"]
                    y_sent = my_dict["sentences"][y_sent_id]["_subword_to_ID"]
                    # This guarantees that trigger x is always before trigger y in narrative order

                    context_start_sent_id = max(x_sent_id-1, 0)
                    context_end_sent_id = min(y_sent_id+2, len(my_dict["sentences"]))
                    sent_num += context_end_sent_id - context_start_sent_id

                    if rev_ind[x] < rev_ind[y]:
                        xy = 0
                    else:
                        xy = 1
                    relations.append(xy)
                    relation_stats[xy] += 1

                    labels_full[instance_id]["event_pairs"].append([i, j])

                    """
                    if (context_end_sent_id - context_start_sent_id) > 5:
                        labels_full[art_id].append(None)
                        continue
                    else:
                        labels_full[art_id].append(xy)
                    """
                    if params['td'] == 1:
                        x_sent, offset_x, new_start_x, new_end_x = add_tense_info(x_sent, my_dict["event_dict"][x]['tense'], my_dict['event_dict'][x]['_subword_id'], my_dict["event_dict"][x]['mention'], 2589, 1736)
                    else:
                        x_sent, offset_x, new_start_x, new_end_x = x_sent, 0, my_dict['event_dict'][x]['_subword_id'], my_dict['event_dict'][x]['_subword_id'] + len(tokenizer.encode(my_dict["event_dict"][x]['mention'])) - 2

                    if x_sent_id != y_sent_id:
                        if params['td'] == 1:
                            y_sent, offset_y, new_start_y, new_end_y = add_tense_info(y_sent, my_dict["event_dict"][y]['tense'], my_dict['event_dict'][y]['_subword_id'], my_dict["event_dict"][y]['mention'], 1404, 5400)
                        else:
                            y_sent, offset_y, new_start_y, new_end_y = y_sent, 0, my_dict['event_dict'][y]['_subword_id'], my_dict['event_dict'][y]['_subword_id'] + len(tokenizer.encode(my_dict["event_dict"][y]['mention'])) - 2
                        for sid in range(context_start_sent_id, context_end_sent_id):
                            if sid == x_sent_id:
                                event_pos.append(new_start_x + len(TokenIDs) - 1)
                                event_pos_end.append(new_end_x + len(TokenIDs) - 1)
                                TokenIDs += x_sent[1:]
                            elif sid == y_sent_id:
                                event_pos.append(new_start_y + len(TokenIDs) - 1)
                                event_pos_end.append(new_end_y + len(TokenIDs) - 1)
                                TokenIDs += y_sent[1:]
                            else:
                                TokenIDs += my_dict["sentences"][sid]["_subword_to_ID"][1:]
                    else:
                        if params['td'] == 1:
                            y_sent, offset_y, new_start_y, new_end_y = add_tense_info(x_sent, my_dict["event_dict"][y]['tense'], my_dict['event_dict'][y]['_subword_id'] + offset_x, my_dict["event_dict"][y]['mention'], 1404, 5400)
                        else:
                            y_sent, offset_y, new_start_y, new_end_y = y_sent, 0, my_dict['event_dict'][y]['_subword_id'], my_dict['event_dict'][y]['_subword_id'] + len(tokenizer.encode(my_dict["event_dict"][y]['mention'])) - 2
                        for sid in range(context_start_sent_id, context_end_sent_id):
                            if sid == y_sent_id:
                                event_pos.append(new_start_x + len(TokenIDs) - 1)
                                event_pos_end.append(new_end_x + len(TokenIDs) - 1)
                                event_pos.append(new_start_y + len(TokenIDs) - 1)
                                event_pos_end.append(new_end_y + len(TokenIDs) - 1)
                                TokenIDs += y_sent[1:]
                            else:
                                TokenIDs += my_dict["sentences"][sid]["_subword_to_ID"][1:]

                    if len(TokenIDs) > max_len:
                        max_len = len(TokenIDs)
                    if pair_num < 5:
                        print("first event:", tokenizer.decode(TokenIDs[event_pos[0]:event_pos_end[0]]))
                        print("second event:", tokenizer.decode(TokenIDs[event_pos[1]:event_pos_end[1]]))
                        print("TokenIDs:", tokenizer.decode(TokenIDs))
                    feature = {'input_ids': TokenIDs,
                               'event_pos': event_pos,
                               'event_pos_end': event_pos_end,
                               'event_pair': [[1, 2]],
                               'labels': relations,
                              }
                    features.append(feature)
                    labels.append(xy)
                
    print(len(labels_full))
    print(len(labels))
    print("Temporal Relation Stats:", relation_stats)
    print("Total num of pairs:", pair_num)
    print("Max length of context:", max_len)
    print("Avg num of sentences that context contains:", sent_num/pair_num)
    return features, labels, labels_full
    
#### TDDiscourse PROCESSING ENDS HERE ####

######################
### FOR PREDICTION ###
######################

if params['testdata'] == "PRED":
    folder = sys.argv[2]
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    art_split = {}
    features_test = []
    for this_file in onlyfiles:
        f = open(folder + '/' + this_file)
    #with open(folder+'/'+onlyfiles[0]) as f:
        data = json.load(f)
        f.close()
        view_map = {}
        count = 0
        for view in data['views']:
            view_map[view['viewName']] = count
            count += 1
           
        event_pos = []
        event_pos_end = []
        char_ = tokenized_to_origin_span(data['text'], data['tokens'])
        if 'SRL_VERB' not in view_map.keys():
            continue
        for constituent in data['views'][view_map['SRL_VERB']]['viewData'][0]['constituents']:
            if constituent['label'] == 'Predicate':
                event_pos.append(char_[constituent['start']][0])
                event_pos_end.append(char_[constituent['start']][1]+1)

        art_dict = {'id': folder+ "/" + this_file, 
                    'text': data['text'],
                    'event_pos': event_pos,
                    'event_pos_end': event_pos_end
                   }
            
        #input_list = [doc_dict]
        pair_num = 0
        #art_split = {}
        #features_test = []
        #for art_dict in input_list:
        text, ep, epe = art_dict['text'], art_dict['event_pos'], art_dict['event_pos_end']
        try:
            my_dict = tdd_reader(text, ep, epe, tokenizer)
        except TypeError:
            print('document id:' + folder+ "/" + this_file)
            continue 
        pairs = []
        for i in range(0, len(ep)):
            for j in range(i+1, len(ep)):
                pair_num += 1
                event_pos = []
                event_pos_end = []
                relations = []
                TokenIDs = [65]
                x, y = i, j
                x_sent_id = my_dict["event_dict"][i]["sent_id"]
                y_sent_id = my_dict["event_dict"][j]["sent_id"]
                if x_sent_id > y_sent_id:
                    x, y = j, i
                    x_sent_id = my_dict["event_dict"][x]["sent_id"]
                    y_sent_id = my_dict["event_dict"][y]["sent_id"]
                elif x_sent_id == y_sent_id:
                    x_position = my_dict["event_dict"][x]["_subword_id"]
                    y_position = my_dict["event_dict"][y]["_subword_id"]
                    if x_position > y_position:
                        x, y = j, i
                else:
                    x, y = i, j
                    
                pairs.append([x, y])
                x_sent = my_dict["sentences"][x_sent_id]["_subword_to_ID"]
                y_sent = my_dict["sentences"][y_sent_id]["_subword_to_ID"]
                # This guarantees that trigger x is always before trigger y in narrative order

                context_start_sent_id = max(x_sent_id-1, 0)
                context_end_sent_id = min(y_sent_id+2, len(my_dict["sentences"]))
                #sent_num += context_end_sent_id - context_start_sent_id
                    
                relations.append(0)

                if params['td'] == 1:
                    x_sent, offset_x, new_start_x, new_end_x = add_tense_info(x_sent, my_dict["event_dict"][x]['tense'], my_dict['event_dict'][x]['_subword_id'], my_dict["event_dict"][x]['mention'], 2589, 1736)
                else:
                    x_sent, offset_x, new_start_x, new_end_x = x_sent, 0, my_dict['event_dict'][x]['_subword_id'], my_dict['event_dict'][x]['_subword_id'] + len(tokenizer.encode(my_dict["event_dict"][x]['mention'])) - 2

                if x_sent_id != y_sent_id:
                    if params['td'] == 1:
                        y_sent, offset_y, new_start_y, new_end_y = add_tense_info(y_sent, my_dict["event_dict"][y]['tense'], my_dict['event_dict'][y]['_subword_id'], my_dict["event_dict"][y]['mention'], 1404, 5400)
                    else:
                        y_sent, offset_y, new_start_y, new_end_y = y_sent, 0, my_dict['event_dict'][y]['_subword_id'], my_dict['event_dict'][y]['_subword_id'] + len(tokenizer.encode(my_dict["event_dict"][y]['mention'])) - 2
                    for sid in range(context_start_sent_id, context_end_sent_id):
                        if sid == x_sent_id:
                            event_pos.append(new_start_x + len(TokenIDs) - 1)
                            event_pos_end.append(new_end_x + len(TokenIDs) - 1)
                            TokenIDs += x_sent[1:]
                        elif sid == y_sent_id:
                            event_pos.append(new_start_y + len(TokenIDs) - 1)
                            event_pos_end.append(new_end_y + len(TokenIDs) - 1)
                            TokenIDs += y_sent[1:]
                        else:
                            TokenIDs += my_dict["sentences"][sid]["_subword_to_ID"][1:]
                else:
                    if params['td'] == 1:
                        y_sent, offset_y, new_start_y, new_end_y = add_tense_info(x_sent, my_dict["event_dict"][y]['tense'], my_dict['event_dict'][y]['_subword_id'] + offset_x, my_dict["event_dict"][y]['mention'], 1404, 5400)
                    else:
                        y_sent, offset_y, new_start_y, new_end_y = y_sent, 0, my_dict['event_dict'][y]['_subword_id'], my_dict['event_dict'][y]['_subword_id'] + len(tokenizer.encode(my_dict["event_dict"][y]['mention'])) - 2
                    for sid in range(context_start_sent_id, context_end_sent_id):
                        if sid == y_sent_id:
                            event_pos.append(new_start_x + len(TokenIDs) - 1)
                            event_pos_end.append(new_end_x + len(TokenIDs) - 1)
                            event_pos.append(new_start_y + len(TokenIDs) - 1)
                            event_pos_end.append(new_end_y + len(TokenIDs) - 1)
                            TokenIDs += y_sent[1:]
                        else:
                            TokenIDs += my_dict["sentences"][sid]["_subword_to_ID"][1:]

                if pair_num < 5:
                    print("first event:", tokenizer.decode(TokenIDs[event_pos[0]:event_pos_end[0]]))
                    print("second event:", tokenizer.decode(TokenIDs[event_pos[1]:event_pos_end[1]]))
                    print("TokenIDs:", tokenizer.decode(TokenIDs))
                feature = {'input_ids': TokenIDs,
                           'event_pos': event_pos,
                           'event_pos_end': event_pos_end,
                           'event_pair': [[1, 2]],
                           'labels': relations,
                          }
                features_test.append(feature)
        #art_split.append(pairs) 
        art_split[art_dict['id']] = pairs

if debug:
    train_dataloader = DataLoader(features_train, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn, drop_last=True)
    valid_dataloader = test_dataloader = train_dataloader
    for step, batch in enumerate(train_dataloader):
        print(batch)
elif params['testdata'] == 'TDD':
    features_valid, labels_valid, labels_full_valid = TDD_processor('man-dev')
    features_test, labels_test, labels_full_test = TDD_processor('man-test')
    #print(len(labels_test)) # man_test, 846
    #print(abnormal_articles) # CNN19980213.2130.0155
    #with open("tdd_labels.json", 'w') as f:
    #    json.dump(labels_full_test, f)
    #with open("tdd_labels.txt", 'w') as f:
    #    for label in labels_test:
    #        print(label, file = f)
    valid_dataloader = DataLoader(features_valid, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn, drop_last=False) 
    test_dataloader = DataLoader(features_test, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn, drop_last=False)
elif params['testdata'] == 'PRED':
    test_dataloader = DataLoader(features_test, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn, drop_last=False)
else:
    train_dataloader = DataLoader(features_train, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn, drop_last=True)
    valid_dataloader = DataLoader(features_valid, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn, drop_last=False)
    test_dataloader = DataLoader(features_test, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn, drop_last=False)
print("  Data processing took: {:}".format(format_time(time.time() - t0)))

OnePassModel = transformers_mlp_cons(params)
OnePassModel.to(cuda)
OnePassModel.zero_grad()
print("# of parameters:", count_parameters(OnePassModel))
    

# PREDICTING MODE
best_F1 = 0.0
best_1 = -0.4
best_2 = 0.6
mem_exp_test = exp(cuda, OnePassModel, params['epochs'], params['learning_rate'], None, None, test_dataloader, params['dataset'], best_PATH, None, params['dpn'], model_name, None, [best_1, best_2])
json_file = 'prediction/' + params['testdata'] + model_name + '.json'
flag, F1 = mem_exp_test.evaluate(eval_data = params['dataset'], test = True, predict = json_file, f1_metric = f1_metric)
with open(json_file, 'r') as f:
    pred = json.load(f)
    logits = []
    for i in pred['array'][1:]:
        logits.append(func(i))
logits = np.array(logits)
temperature = 0.444
scaled_model = ModelWithTemperature(temperature)
scaled_logits = scaled_model(logits)
logits = scaled_logits.cpu().detach().numpy() 
logits = softmax(logits)

predict_with_abstention = []
for i in logits:
    if entropy(i) < 0.402 and max(i) > 0.892:
        predict_with_abstention.append(1)
    else:
        predict_with_abstention.append(0)

with open('prediction/' + params['testdata'] + model_name + 'lambda_1' + str(best_1) + 'lambda_2' + str(best_2) + 'temperature' + str(temperature) + '.npy', 'wb') as f:
    np.save(f, logits)

output_timeline = tl_construction(logits, art_split, long = True, softmax = False, predict_with_abstention = predict_with_abstention)

with open(sys.argv[3] + '/' + sys.argv[2].split('/')[-1] + '_output.json', 'w') as f:
    json.dump(output_timeline, f)

