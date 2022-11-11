import json
import os
from os import listdir
from os.path import isfile, join
import time
import datetime
from datetime import datetime 
import random
from document_reader import *
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import transformers_mlp_cons
from exp import *
import numpy as np
import pickle
import csv
from pprint import pprint

mask_in_input_ids = False
mask_in_input_mask = not mask_in_input_ids

# datetime object containing current date and time
now = datetime.datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
print("date and time =", dt_string)

#label_dict={"SuperSub": 0, "SubSuper": 1, "Coref": 2, "NoRel": 3}
#num_dict = {0: "before", 1: "after", 2: "equal", 3: "vague"}
#def label_to_num(label):
#    return label_dict[label]
#def num_to_label(num):
#    return num_dict[num]

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
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = [[1.0] * len(f['input_ids']) + [0.0] * (max_len - len(f['input_ids'])) for f in batch]
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    event_pos = [f['event_pos'] for f in batch]    # Transformer tokenized token ids
    event_pos_end = [f['event_pos_end'] for f in batch]
    event_pair = [f['event_pair'] for f in batch]
    labels = [f['labels'] for f in batch]
    output = (input_ids, input_mask, event_pos, event_pos_end, event_pair, labels)
    return output

def collate_fn_mask(batch):
    max_len = max([len(f['input_ids']) for f in batch])
    input_ids = [f['input_ids'] + [0] * (max_len - len(f['input_ids'])) for f in batch]
    if mask_in_input_ids:
        for f_id, token_ids in enumerate(input_ids):
            for event_p in batch[f_id]['event_pos']:
                token_ids[event_p] = 67    # [MASK]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = [[1.0] * len(f['input_ids']) + [0.0] * (max_len - len(f['input_ids'])) for f in batch]
    if mask_in_input_mask:
        for f_id, token_ids in enumerate(input_ids):
            for event_p in batch[f_id]['event_pos']:
                token_ids[event_p] = 0.0    # <pad>
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    event_pos = [f['event_pos'] for f in batch]
    event_pos_end = [f['event_pos_end'] for f in batch]
    event_pair = [f['event_pair'] for f in batch]
    labels = [f['labels'] for f in batch]
    output = (input_ids, input_mask, event_pos, event_pos_end, event_pair, labels)
    return output

def bodygraph_processor(text):
    event_pos_char = []
    event_pos_end_char = []
    text = text.replace("<p>", "")
    text = text.replace("</p>", "")
    
    result = text.find("<span style=\'color:")
    while result != -1:
        if text[result+19] == "r":    # red
            offset = 33
        else:                         # blue
            offset = 34
        text = text[0:result] + text[(result+offset):]
        event_pos_char.append(result)
        end = text.find("</strong></span>")
        event_pos_end_char.append(end-2)
        text = text[0:end] + text[end+16:]
        result = text.find("<span style=\'color:")

    return text, event_pos_char, event_pos_end_char  

def contrast_matres_reader(text, event_pos_char, event_pos_end_char, tokenizer):
    my_dict = {}
    my_dict["event_dict"] = {}
    my_dict["sentences"] = []    
    my_dict["doc_content"] = text
    
    my_dict["event_dict"][1] = {"mention": text[event_pos_char[0]:event_pos_end_char[0]+1], 
                                "start_char": event_pos_char[0], 
                                "end_char": event_pos_end_char[0]} 
    my_dict["event_dict"][2] = {"mention": text[event_pos_char[1]:event_pos_end_char[1]+1], 
                                "start_char": event_pos_char[1], 
                                "end_char": event_pos_end_char[1]} 
    
    # Split document into sentences
    sent_tokenized_text = sent_tokenize(my_dict["doc_content"])
    sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
    count_sent = 0
    end_pos = [1]
    for sent in sent_tokenized_text:
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        
        spacy_token = nlp(sent)
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        # spaCy-tokenized tokens & Part-Of-Speech Tagging
        for token in spacy_token:
            sent_dict["tokens"].append(token.text)
            sent_dict["pos"].append(token.pos_)
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])

        # huggingface tokenizer
        sent_dict["_subword_to_ID"], sent_dict["_subwords"], \
        sent_dict["_subword_span_SENT"], sent_dict["_subword_map"] = \
        transformers_list(sent_dict["content"], tokenizer, sent_dict["tokens"], sent_dict["token_span_SENT"])
        
        if count_sent == 0:
            end_pos.append(len(sent_dict["_subword_to_ID"]))
        else:
            end_pos.append(end_pos[-1] + len(sent_dict["_subword_to_ID"]) - 1)
            
        sent_dict["_subword_span_DOC"] = \
        span_SENT_to_DOC(sent_dict["_subword_span_SENT"], sent_dict["sent_start_char"])
        
        sent_dict["_subword_pos"] = []
        for token_id in sent_dict["_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["_subword_pos"].append("None")
            else:
                sent_dict["_subword_pos"].append(sent_dict["pos"][token_id])
        
        my_dict["sentences"].append(sent_dict)
        count_sent += 1
        
    my_dict['end_pos'] = end_pos
    # Add sent_id as an attribute of event
    for event_id, event_dict in my_dict["event_dict"].items():
        my_dict["event_dict"][event_id]["sent_id"] = sent_id = \
        sent_id_lookup(my_dict, event_dict["start_char"], event_dict["end_char"])
        my_dict["event_dict"][event_id]["token_id"] = \
        id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"])
        my_dict["event_dict"][event_id]["_subword_id"] = \
        id_lookup(my_dict["sentences"][sent_id]["_subword_span_DOC"], event_dict["start_char"]) + 1
        # updated on Mar 20, 2021, plus 1 because of [CLS] or <s>

    return my_dict

params = {'transformers_model': 'google/bigbird-roberta-large',
          'dataset': 'MATRES',   # 'HiEve', 'IC', 'MATRES' 
          'block_size': 64,
          'add_loss': 0, 
          'batch_size': 1,    # 6 works on 48G gpu
          'epochs': 30,
          'learning_rate': 3e-06,    # subject to change
          'seed': 42,
          'gpu_id': '1',    # subject to change
          'debug': 0,
          #'rst_file_name': "0111-lr5e-6-b1-gpu6-loss0-dataMATRES-accum1.rst",    # exp-4893
          #'rst_file_name': "0204-lr5e-6-b3-gpu3-loss0-dataMATRES-accum1.rst", 
          'rst_file_name': "0204-lr5e-6-b1-gpu5-loss0-dataMATRES-accum1.rst",
          'mask_in_input_ids': mask_in_input_ids,
          'mask_in_input_mask': mask_in_input_mask,
         }
if params['transformers_model'][-5:] == "large":
    params['emb_size'] = 1024
elif params['transformers_model'][-4:] == "base":
    params['emb_size'] = 768
else:
    print("Something weird happens...")
    
set_seed(params['seed'])
rst_file_name = params['rst_file_name']
model_params_dir = "./model_params/"
if params['dataset'] == 'HiEve':
    best_PATH = model_params_dir + "HiEve_best/" + rst_file_name.replace(".rst", ".pt") # to save model params here
elif params['dataset'] == 'IC':
    best_PATH = model_params_dir + "IC_best/" + rst_file_name.replace(".rst", ".pt") # to save model params here
elif params['dataset'] == 'MATRES':
    best_PATH = model_params_dir + "MATRES_best/" + rst_file_name.replace(".rst", ".pt") # to save model params here
os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu_id']
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
cuda = torch.device('cuda')
params['cuda'] = cuda # not included in config file

model = transformers_mlp_cons(params)
model.to(cuda)
#model.zero_grad()
print("# of parameters:", count_parameters(model))
#for name, param in model.named_parameters():
#    if param.requires_grad:
#        print(name, param.data.size())
model_name = rst_file_name.replace(".rst", "") # to be designated after finding the best parameters
tokenizer = AutoTokenizer.from_pretrained(params['transformers_model']) 

contrast_set_csv = "contrast-sets/MATRES/Platinum_subset_minimal_pairs.csv"
original = []
contrast = []
orig_labels = []
contrast_labels = []
original_id = set()
label_map = {'before': 0, 'after': 1, 'simultaneous': 2, 'vague': 3}
with open(contrast_set_csv) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[-1] != "reason":
            if row[0] not in original_id and row[2] != '':
                original_id.add(row[0])
                text, start, end = bodygraph_processor(row[2])
                my_dict = contrast_matres_reader(text, start, end, tokenizer)
                TokenIDs = docTransformerTokenIDs(my_dict['sentences'])
                event_pos = []
                event_pos_end = []
                for event_id in my_dict['event_dict'].keys():
                    sent_id = my_dict['event_dict'][event_id]['sent_id']
                    start = my_dict['end_pos'][sent_id] - 1 + my_dict['event_dict'][event_id]['_subword_id'] # 0 x x x x 0 x x x 0
                    event_pos.append(start)    
                    subword_len = len(tokenizer.encode(my_dict["event_dict"][event_id]['mention'])) - 2
                    event_pos_end.append(start + subword_len)
                feature = {'input_ids': TokenIDs,
                           'event_pos': event_pos,
                           'event_pos_end': event_pos_end,
                           'event_pair': [[1, 2]],
                           'labels': [label_map[row[4].strip()]],
                          }
                original.append(feature)
                orig_labels.append(label_map[row[4].strip()])
            if row[5] != '':
                text, start, end = bodygraph_processor(row[5])
                my_dict = contrast_matres_reader(text, start, end, tokenizer)
                TokenIDs = docTransformerTokenIDs(my_dict['sentences'])
                event_pos = []
                event_pos_end = []
                for event_id in my_dict['event_dict'].keys():
                    sent_id = my_dict['event_dict'][event_id]['sent_id']
                    start = my_dict['end_pos'][sent_id] - 1 + my_dict['event_dict'][event_id]['_subword_id'] # 0 x x x x 0 x x x 0
                    event_pos.append(start)    
                    subword_len = len(tokenizer.encode(my_dict["event_dict"][event_id]['mention'])) - 2
                    event_pos_end.append(start + subword_len)
                feature = {'input_ids': TokenIDs,
                            'event_pos': event_pos,
                            'event_pos_end': event_pos_end,
                            'event_pair': [[1, 2]],
                            'labels': [label_map[row[6].strip()]],
                            }
                contrast.append(feature)
                contrast_labels.append(label_map[row[6].strip()])
                
c = False    # if False -> original; if True -> contrast
if c:
    testset = contrast
    labels = contrast_labels
else:
    testset = original
    labels = orig_labels
mask = False    # Don't change this, since masking is not working at all. -- Haoyu on 02/18/2022
if mask:
    test_dataloader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=collate_fn_mask, drop_last=False)
else:
    test_dataloader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)
mem_exp = exp(cuda, model, params['epochs'], params['learning_rate'], None, None, test_dataloader, params['dataset'], best_PATH, None, model_name)
mem_exp.evaluate(eval_data = params['dataset'], test = True, predict = 'prediction/' + model_name + '.json')

with open('prediction/' + model_name + '.json') as f:
    logits = json.load(f)
    results = logits['array'][1:]
    correct = 0
    pair_count = -1
    for instance_id, pred in enumerate(results):
        pair_count += 1
        pred = np.array(pred)
        if np.argmax(pred) == labels[instance_id]:
            correct += 1
    print("Acc:")
    print(correct / pair_count)