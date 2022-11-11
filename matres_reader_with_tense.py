from transformers import AutoTokenizer
import os
from os import listdir
from os.path import isfile, join
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")
from eventseg_getter import *
from tense_tagger import *
import requests

#def tense_getter(sentence):
#    headers = {'Content-type':'application/json'}
#    response = requests.post('https://tense-sense-identifier.herokuapp.com/home', json={"data": sentence}, headers=headers)
#    if response.status_code != 200:
#        print("tense_response:", response.status_code)
#    result = json.loads(response.text)
#    return result
def tense_getter(txt):
    return print_parsed_text(txt)

space = ' '

#model = RobertaModel.from_pretrained('roberta-base')
#dir_name = "/shared/why16gzl/logic_driven/Quizlet/Quizlet_2/LDC2020E20_KAIROS_Quizlet_2_TA2_Source_Data_V1.0/data/ltf/ltf/"
#file_name = "K0C03N4LR.ltf.xml"    # Use ltf_reader 
#dir_name = "/home1/w/why16gzl/KAIROS/hievents_v2/processed/"
#file_name = "article-10901.tsvx"   # Use tsvx_reader

# ============================
#         PoS Tagging
# ============================
pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
identity_matrix = np.identity(len(pos_tags))
postag_to_OneHot = {}
postag_to_OneHot["None"] = np.zeros(len(pos_tags))
for (index, item) in enumerate(pos_tags):
    postag_to_OneHot[item] = identity_matrix[index]
    
def postag_2_OneHot(postag):
    return postag_to_OneHot[postag]

# ===========================
#        HiEve Labels
# ===========================

label_dict={"SuperSub": 0, "SubSuper": 1, "Coref": 2, "NoRel": 3}
num_dict = {0: "SuperSub", 1: "SubSuper", 2: "Coref", 3: "NoRel"}
def label_to_num(label):
    return label_dict[label]
def num_to_label(num):
    return num_dict[num]

# Padding function, both for huggingface encoded sentences, and for part-of-speech tags
def padding(sent, pos = False, max_sent_len = 200):
    if pos == False:
        one_list = [1] * max_sent_len
        one_list[0:len(sent)] = sent
        return torch.tensor(one_list, dtype=torch.long)
    else:
        one_list = ["None"] * max_sent_len
        one_list[0:len(sent)] = sent
        return one_list

def transformers_list(content, tokenizer, token_list = None, token_span_SENT = None):
    #tokenizer = AutoTokenizer.from_pretrained(transformers_model)    
    encoded = tokenizer.encode(content)
    # input_ids = torch.tensor(encoded).unsqueeze(0)  # Batch size 1
    # outputs = model(input_ids)
    # last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    _subwords = []
    _subword_to_ID = []
    _subwords_no_space = []
    for index, i in enumerate(encoded):
        r_token = tokenizer.decode([i])
        if len(r_token) > 0:
            _subword_to_ID.append(i)
            _subwords.append(r_token)
            if r_token[0] == " ":
                _subwords_no_space.append(r_token[1:])
            else:
                _subwords_no_space.append(r_token)

    _subword_span = tokenized_to_origin_span(content, _subwords_no_space[1:-1]) # w/o <s> and </s>
    _subword_map = []
    if token_span_SENT is not None:
        _subword_map.append(-1) # "<s>"
        for subword in _subword_span:
            _subword_map.append(token_id_lookup(token_span_SENT, subword[0], subword[1]))
        _subword_map.append(-1) # "</s>" 
        return _subword_to_ID, _subwords, _subword_span, _subword_map
    else:
        return _subword_to_ID, _subwords, _subword_span, -1

def tokenized_to_origin_span(text, token_list):
    token_span = []
    pointer = 0
    previous_pointer = 0
    for token in token_list:
        while pointer < len(text):
            if token[0] == text[pointer]:
                start = pointer
                end = start + len(token) - 1
                previous_pointer = pointer = end + 1
                break
            else:
                pointer += 1
        if pointer < len(text):
            token_span.append([start, end])
        else:
            if previous_pointer < len(text):
                # exceeding text length, meaning that a weird character is encountered
                token_span.append([end + 1, end + 1])
                pointer = previous_pointer
            else:
                # end of text
                token_span.append([start, end])
    return token_span

def sent_id_lookup(my_dict, start_char, end_char = None):
    for sent_dict in my_dict['sentences']:
        if end_char is None:
            if start_char >= sent_dict['sent_start_char'] and start_char <= sent_dict['sent_end_char']:
                return sent_dict['sent_id']
        else:
            if start_char >= sent_dict['sent_start_char'] and end_char <= sent_dict['sent_end_char']:
                return sent_dict['sent_id']

def token_id_lookup(token_span_SENT, start_char, end_char):
    for index, token_span in enumerate(token_span_SENT):
        if start_char >= token_span[0] and end_char <= token_span[1]:
            return index

def span_SENT_to_DOC(token_span_SENT, sent_start):
    token_span_DOC = []
    #token_count = 0
    for token_span in token_span_SENT:
        start_char = token_span[0] + sent_start
        end_char = token_span[1] + sent_start
        #assert my_dict["doc_content"][start_char] == sent_dict["tokens"][token_count][0]
        token_span_DOC.append([start_char, end_char])
        #token_count += 1
    return token_span_DOC

def id_lookup(span_SENT, start_char):
    # this function is applicable to huggingface subword or token from ltf/spaCy
    # id: start from 0
    token_id = -1
    for token_span in span_SENT:
        token_id += 1
        if token_span[0] <= start_char and token_span[1] >= start_char:
            return token_id
    raise ValueError("Nothing is found.")
    return token_id

def segment_id_lookup(segments, sent_id):
    for i in range(len(segments)):
        if sent_id > segments[i] and sent_id <= segments[i+1]:
            return i
"""
map_tense = {"futu": "Future",
             "perf": "Perfect",
             "simp": "Simple", 
             "cont": "Continuous",
             "past": "Past",
             "pres": "Present",
             "pass": "Passive"
            }
def tense_mapper(tense):
    tense = list(map(''.join, zip(*[iter(tense)]*4))) 
    return ' '.join([map_tense[i] for i in tense])
"""
map_tense = {"futu": "Future",
             "perf": "Perfect",
             #"simp": "",      # DELETE simp
             "simp": "Simple", # KEEP simp
             "cont": "Continuous",
             "past": "Past",
             "pres": "Present",
             "pass": "Passive",
            }
def tense_mapper(tense):
    tense = list(map(''.join, zip(*[iter(tense)]*4))) 
    return ' '.join([map_tense[i] for i in tense]).strip()
        
def tense_finder(tense_list, start_char):
    # start_char: sentence level 
    # e.g., 'She would say the soldiers were hit by a truck.'
    # ['would', 'MD', [4, 9], 'futusimp']
    # ['say', 'VB', [10, 13], 'futusimp', ...
    for verb in tense_list:
        if verb[2][0] == start_char:
            try:
                return [tense_mapper(verb[3]), '['+verb[3]+']'] 
            # Updated on Oct 29, 2022, because of unknown error
            except:
                return None
            # Updated on Feb 27, 2022
    #print(start_char)
    #print(tense_list)
    return None
# ========================================
#        MATRES: read relation file
# ========================================
# MATRES has separate text files and relation files
# We first read relation files
"""
mypath_TB = './MATRES/TBAQ-cleaned/TimeBank/' # after correction
onlyfiles_TB = [f for f in listdir(mypath_TB) if isfile(join(mypath_TB, f))]
mypath_AQ = './MATRES/TBAQ-cleaned/AQUAINT/' 
onlyfiles_AQ = [f for f in listdir(mypath_AQ) if isfile(join(mypath_AQ, f))]
mypath_PL = './MATRES/te3-platinum/'
onlyfiles_PL = [f for f in listdir(mypath_PL) if isfile(join(mypath_PL, f))]
MATRES_timebank = './MATRES/timebank.txt'
MATRES_aquaint = './MATRES/aquaint.txt'
MATRES_platinum = './MATRES/platinum.txt'
temp_label_map = {"BEFORE": 0, "AFTER": 1, "EQUAL": 2, "VAGUE": 3}
eiid_to_event_trigger = {}
eiid_pair_to_label = {}   

# =========================
#       MATRES Reader
# =========================
def MATRES_READER(matres_file, eiid_to_event_trigger, eiid_pair_to_label):
    with open(matres_file, "r") as f_matres:
        content = f_matres.read().split("\n")
        for rel in content:
            rel = rel.split("\t")
            fname = rel[0]
            trigger1 = rel[1]
            trigger2 = rel[2]
            eiid1 = int(rel[3])
            eiid2 = int(rel[4])
            tempRel = temp_label_map[rel[5]]

            if fname not in eiid_to_event_trigger:
                eiid_to_event_trigger[fname] = {}
                eiid_pair_to_label[fname] = {}
            eiid_pair_to_label[fname][(eiid1, eiid2)] = tempRel
            if eiid1 not in eiid_to_event_trigger[fname].keys():
                eiid_to_event_trigger[fname][eiid1] = trigger1
            if eiid2 not in eiid_to_event_trigger[fname].keys():
                eiid_to_event_trigger[fname][eiid2] = trigger2

MATRES_READER(MATRES_timebank, eiid_to_event_trigger, eiid_pair_to_label)
MATRES_READER(MATRES_aquaint, eiid_to_event_trigger, eiid_pair_to_label)
MATRES_READER(MATRES_platinum, eiid_to_event_trigger, eiid_pair_to_label)

def tml_reader(dir_name, file_name, tokenizer):
    my_dict = {}
    my_dict["event_dict"] = {}
    my_dict["eiid_dict"] = {}
    my_dict["doc_id"] = file_name.replace(".tml", "") 
    # e.g., file_name = "ABC19980108.1830.0711.tml"
    # dir_name = '/shared/why16gzl/logic_driven/EMNLP-2020/MATRES/TBAQ-cleaned/TimeBank/'
    tree = ET.parse(dir_name + file_name)
    root = tree.getroot()
    MY_STRING = str(ET.tostring(root))
    # ================================================
    # Load the lines involving event information first
    # ================================================
    event_id_why = 0
    for makeinstance in root.findall('MAKEINSTANCE'):
        instance_str = str(ET.tostring(makeinstance)).split(" ")
        try:
            assert instance_str[3].split("=")[0] == "eventID"
            assert instance_str[2].split("=")[0] == "eiid"
            eiid = int(instance_str[2].split("=")[1].replace("\"", "")[2:])
            eID = instance_str[3].split("=")[1].replace("\"", "")
        except:
            for i in instance_str:
                if i.split("=")[0] == "eventID":
                    eID = i.split("=")[1].replace("\"", "")
                if i.split("=")[0] == "eiid":
                    eiid = int(i.split("=")[1].replace("\"", "")[2:])
        # Not all document in the dataset contributes relation pairs in MATRES
        # Not all events in a document constitute relation pairs in MATRES
        
        if my_dict["doc_id"] in eiid_to_event_trigger.keys():
            if eiid in eiid_to_event_trigger[my_dict["doc_id"]].keys():
                event_id_why += 1
                my_dict["event_dict"][eID] = {"eiid": eiid, "mention": eiid_to_event_trigger[my_dict["doc_id"]][eiid], "event_id_why": event_id_why}
                my_dict["eiid_dict"][eiid] = {"eID": eID}
        
    # ==================================
    #              Load Text
    # ==================================
    start = MY_STRING.find("<TEXT>") + 6
    end = MY_STRING.find("</TEXT>")
    MY_TEXT = MY_STRING[start:end]
    while MY_TEXT[0] == " ":
        MY_TEXT = MY_TEXT[1:]
    MY_TEXT = MY_TEXT.replace("\\n", " ")
    MY_TEXT = MY_TEXT.replace("\\'", "\'")
    MY_TEXT = MY_TEXT.replace("  ", " ")
    MY_TEXT = MY_TEXT.replace(" ...", "...")
    
    # ========================================================
    #    Load position of events, in the meantime replacing 
    #    "<EVENT eid="e1" class="OCCURRENCE">turning</EVENT>"
    #    with "turning"
    # ========================================================
    while MY_TEXT.find("<") != -1:
        start = MY_TEXT.find("<")
        end = MY_TEXT.find(">")
        if MY_TEXT[start + 1] == "E":
            event_description = MY_TEXT[start:end].split(" ")
            eID = (event_description[2].split("="))[1].replace("\"", "")
            MY_TEXT = MY_TEXT[:start] + MY_TEXT[(end + 1):]
            if eID in my_dict["event_dict"].keys():
                my_dict["event_dict"][eID]["start_char"] = start # loading position of events
        else:
            MY_TEXT = MY_TEXT[:start] + MY_TEXT[(end + 1):]
    
    # =====================================
    # Enter the routine for text processing
    # =====================================
    
    my_dict["doc_content"] = MY_TEXT
    my_dict["sentences"] = []
    my_dict["relation_dict"] = {}
    sent_tokenized_text = sent_tokenize(my_dict["doc_content"])
    #sent_tokenized_text = []
    #tense_res = tense_getter(MY_TEXT)
    #for sentence in tense_res['sentences']:
    #    sent_tokenized_text.append(sentence[0])
        
    sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
    end_pos = [1]
    for count_sent, sent in enumerate(sent_tokenized_text):
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        sent_dict["tense_list"] = tense_getter(sent)
        
        spacy_token = nlp(sent_dict["content"])
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
        sent_id_lookup(my_dict, event_dict["start_char"])
        my_dict["event_dict"][event_id]["token_id"] = \
        id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"])
        my_dict["event_dict"][event_id]["_subword_id"] = \
        id_lookup(my_dict["sentences"][sent_id]["_subword_span_DOC"], event_dict["start_char"]) + 1 
        # updated on Mar 20, 2021
        #my_dict["event_dict"][event_id]["tense"] = tense_finder(tense_res['sentences'][sent_id][1], event_dict["start_char"] - my_dict['sentences'][sent_id]["sent_start_char"])
        # updated on Feb 21, 2021
        my_dict["event_dict"][event_id]["tense"] = tense_finder(my_dict['sentences'][sent_id]["tense_list"], event_dict["start_char"] - my_dict['sentences'][sent_id]["sent_start_char"])
        # updated on Oct 24, 2022, because of change of tense identification service
        
    return my_dict
"""
def tdd_reader(text, event_pos, event_pos_end, tokenizer):
    my_dict = {}
    my_dict["event_dict"] = {}
    my_dict["sentences"] = []
    my_dict["relation_dict"] = {}
    
    # Read tsvx file
    my_dict["doc_content"] = text
        
    for i, pos in enumerate(event_pos):
        my_dict["event_dict"][i] = {"mention": text[event_pos[i]:event_pos_end[i]], "start_char": event_pos[i], "end_char": event_pos_end[i]} 
    
    # Split document into sentences
    sent_tokenized_text = sent_tokenize(my_dict["doc_content"])
    #sent_tokenized_text = []
    #tense_res = tense_getter(text)
    #for sentence in tense_res['sentences']:
    #    sent_tokenized_text.append(sentence[0])
        
    sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
    count_sent = 0
    end_pos = [1]
    for sent in sent_tokenized_text:
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        sent_dict["tense_list"] = tense_getter(sent)
        
        spacy_token = nlp(sent_dict["content"])
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
        #my_dict["event_dict"][event_id]["token_id"] = \
        #id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"]) # updated on Oct 29, 2022, because of unknown error
        my_dict["event_dict"][event_id]["_subword_id"] = \
        id_lookup(my_dict["sentences"][sent_id]["_subword_span_DOC"], event_dict["start_char"]) + 1
        # updated on Mar 20, 2021, plus 1 because of [CLS] or <s>
        my_dict["event_dict"][event_id]["tense"] = tense_finder(my_dict['sentences'][sent_id]["tense_list"], event_dict["start_char"] - my_dict['sentences'][sent_id]["sent_start_char"])
        # updated on Oct 24, 2022, because of change of tense identification service
    return my_dict

def convert_t5_input(text, tl):
    #text = t5_TDD_dic['man-test']['APW19980308.0201'][0][0][19:]
    pointer = text.find("<extra_id_")
    event_pos = []
    event_pos_end = []
    event_ids = []
    e_dict = {}
    while pointer != -1:
        offset = 2
        while text[pointer-offset].isalpha():
            offset += 1
        event_pos.append(pointer-offset+1)
        event_pos_end.append(pointer-1)
        offset = 1
        while text[pointer+10+offset].isdigit():
            offset += 1
        event_id = text[pointer+10:pointer+10+offset]
        event_ids.append(event_id)
        assert text[pointer+10+offset] == '>'
        offset += 1
        if text[pointer+10+offset] == ' ':
            offset += 1
        text = text[0:pointer] + text[pointer+10+offset:]
        pointer = text.find("<extra_id_")
    for i, event_id in enumerate(event_ids):
        e_dict[event_id] = i
    timeline = []
    abnormal = 0
    for i in tl:
        try:
            timeline.append(e_dict[i.replace('<extra_id_', '').replace('>', '')])
        except:
            #print("abnormal")
            abnormal = 1
    return text, event_pos, event_pos_end, event_ids, e_dict, timeline, abnormal





