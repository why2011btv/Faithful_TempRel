
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
