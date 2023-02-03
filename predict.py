import tqdm
import time
import datetime
from datetime import datetime 
import random
from document_reader import *
import os
from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader
from util import *
from pprint import pprint
from transformers import AutoTokenizer
from model import transformers_mlp_cons
from exp import *
import numpy as np
import json
import sys
import pickle

# datetime object containing current date and time
now = datetime.datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
print("date and time =", dt_string)

label_dict={"SuperSub": 0, "SubSuper": 1, "Coref": 2, "NoRel": 3}
num_dict = {0: "SuperSub", 1: "SubSuper", 2: "Coref", 3: "NoRel"}
def label_to_num(label):
    return label_dict[label]
def num_to_label(num):
    return num_dict[num]

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
    event_pos = [f['event_pos'] for f in batch]
    event_pos_end = [f['event_pos_end'] for f in batch]
    event_pair = [f['event_pair'] for f in batch]
    labels = [f['labels'] for f in batch]
    output = (input_ids, input_mask, event_pos, event_pos_end, event_pair, labels)
    return output

def ta_reader(ta, tokenizer):
    my_dict = {}
    my_dict['doc_id'] = ta['corpusId']
    my_dict["event_dict"] = {}
    my_dict["sentences"] = []
    
    sEPs = [0]
    sEPs.extend(ta['sentences']['sentenceEndPositions'])
    end_pos = [1]
    count_sent = 0
    for i in range(len(sEPs) - 1):
        sent_dict = {}
        sent_dict['sent_id'] = i
        sent_dict['tokens'] = ta['tokens'][sEPs[i]:sEPs[i+1]]
        sent_dict['content'] = ' '.join(sent_dict['tokens'])
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent_dict['content'], sent_dict["tokens"])
        
        # huggingface tokenizer
        sent_dict["_subword_to_ID"], sent_dict["_subwords"], \
        sent_dict["_subword_span_SENT"], sent_dict["_subword_map"] = \
        transformers_list(sent_dict["content"], tokenizer, sent_dict["tokens"], sent_dict["token_span_SENT"])
        
        if count_sent == 0:
            end_pos.append(len(sent_dict["_subword_to_ID"]))
        else:
            end_pos.append(end_pos[-1] + len(sent_dict["_subword_to_ID"]) - 1)
        my_dict['sentences'].append(sent_dict)
        count_sent += 1
    my_dict['end_pos'] = end_pos
    
    event_id = 0
    for constituent in ta['views'][0]['viewData'][0]['constituents']:
        event_id += 1
        sent_id = constituent['properties']['sentence_id']
        start_char = my_dict['sentences'][sent_id]['token_span_SENT'][constituent['start_sent_level']][0]
        subword_id = id_lookup(my_dict["sentences"][sent_id]["_subword_span_SENT"], start_char) + 1
        mention = constituent['properties']['predicate'][0]
        my_dict['event_dict'][event_id] = {'mention': mention, '_subword_id': subword_id, 'sent_id': sent_id}
        
    return my_dict
        
def save_IE_output(IE_output, topic):
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    with open('/shared/why16gzl/Repositories/Schema_Induction_with_GPT/IE_output/' + topic + '_' + time_str + '.pkl', 'wb') as f:
        pickle.dump(IE_output, f)
        
#############################
### Setting up parameters ###
#############################

params = {'transformers_model': 'google/bigbird-roberta-base',
          'dataset': 'HiEve',   # 'HiEve', 'IC', 'MATRES' 
          'emb_size': 768,
          'block_size': 64,
          'add_loss': 0, 
          'batch_size': 5,    # 6 works on 48G gpu
          'epochs': 30,
          'learning_rate': 3e-06,    # subject to change
          'seed': 42,
          'gpu_id': '7',    # subject to change
          'debug': 0,
          'rst_file_name': "1116-lr3e-6-b5-gpu6-loss0-dataHiEve.rst",    # subject to change # exp-2778 or exp-3104
         }
set_seed(params['seed'])
rst_file_name = params['rst_file_name']
model_params_dir = "./model_params/"
if params['dataset'] == 'HiEve':
    best_PATH = model_params_dir + "HiEve_best/" + rst_file_name.replace(".rst", ".pt") # to save model params here
else:
    best_PATH = model_params_dir + "IC_best/" + rst_file_name.replace(".rst", ".pt") # to save model params here
os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu_id']
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
cuda = torch.device('cuda')
params['cuda'] = cuda # not included in config file
### Read command line parameters ###
if len(sys.argv) > 1:
    input_file, f_out = sys.argv[1], sys.argv[2]
    
print("Processing input data...")
with open(input_file, "rb") as f:
    IE_output = pickle.load(f)
    
features = []
num_pairs = []
tokenizer = AutoTokenizer.from_pretrained(params['transformers_model'])    
for instance in IE_output:
    my_dict = ta_reader(instance, tokenizer)
    TokenIDs = docTransformerTokenIDs(my_dict['sentences'])
    event_pos = []
    event_pos_end = []
    for event_id in my_dict['event_dict'].keys():
        sent_id = my_dict['event_dict'][event_id]['sent_id']
        start = my_dict['end_pos'][sent_id] - 1 + my_dict['event_dict'][event_id]['_subword_id'] # 0 x x x x 0 x x x 0
        event_pos.append(start)    
        subword_len = len(tokenizer.encode(my_dict["event_dict"][event_id]['mention'])) - 2
        event_pos_end.append(start + subword_len)
        #print(tokenizer.decode([TokenIDs[start]]))
        
    pairs = []
    relations = []
    for i in range(1, len(my_dict['event_dict'])+1):
        for j in range(i+1, len(my_dict['event_dict'])+1):
            pairs.append([i, j])
            relations.append(0)
    num_pairs.append(len(pairs))
    feature = {'input_ids': TokenIDs,
               'event_pos': event_pos,
               'event_pos_end': event_pos_end,
               'event_pair': pairs,
               'labels': relations,
              }
    features.append(feature)
#print(len(features))
    
test_dataloader = DataLoader(features, batch_size=5, shuffle=False, collate_fn=collate_fn, drop_last=False)

model = transformers_mlp_cons(params)
model.to(cuda)
#model.zero_grad()
print("# of parameters:", count_parameters(model))
#for name, param in model.named_parameters():
#    if param.requires_grad:
#        print(name, param.data.size())
model_name = rst_file_name.replace(".rst", "") # to be designated after finding the best parameters

mem_exp = exp(cuda, model, params['epochs'], params['learning_rate'], None, None, test_dataloader, params['dataset'], best_PATH, None, model_name)
mem_exp.evaluate(eval_data = params['dataset'], test = True, predict = 'prediction/' + f_out + model_name + '.json')


with open('prediction/' + f_out + model_name + '.json') as f:
    logits = json.load(f)
preds = []
results = logits['array'][1:]
for i in range(len(num_pairs)):
    if i == 0:
        prev = 0
        preds.append(results[prev:num_pairs[i]])
        prev += num_pairs[i]
    else:
        preds.append(results[prev:prev+num_pairs[i]])
        prev += num_pairs[i]

prediction = []
for doc_preds in preds:
    pair_count = -1
    pc_in_doc = []
    for pred in doc_preds:
        pair_count += 1
        pred = np.array(pred)
        if np.argmax(pred) in [0, 1]:
            pc_in_doc.append([pair_count, np.argmax(pred)])
    prediction.append(pc_in_doc)
    
doc_id = -1    
for instance in IE_output:
    doc_id += 1
    event_num = len(instance['views'][0]['viewData'][0]['constituents'])
    pairs = []
    for i in range(1, event_num+1):
        for j in range(i+1, event_num+1):
            pairs.append([i, j])
    for [pair_count, pred] in prediction[doc_id]:
        src = pairs[pair_count][0] - 1
        tgt = pairs[pair_count][1] - 1
        instance['views'][0]['viewData'][0]['relations'].append({'properties': {'predictor': model_name},
                                                                 'relationName': num_to_label(pred),
                                                                 'srcConstituent': src,
                                                                 'targetConstituent': tgt})
        #print(instance['views'][0]['viewData'][0]['constituents'][src]['properties']['predicate'][0],
        #      instance['views'][0]['viewData'][0]['constituents'][tgt]['properties']['predicate'][0],
        #      num_to_label(pred))

save_IE_output(IE_output, input_file.split('/')[-1].split('_')[0])
        
# datetime object containing current date and time
now = datetime.datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
print("date and time =", dt_string)
