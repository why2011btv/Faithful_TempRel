from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
import string 

punct = string.punctuation 
lemmatizer = WordNetLemmatizer()

freq_lemmas = []
with open('top_5k_freq_lemmas.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        freq_lemmas.append(line[:-1])

def contain_punct(word):
    for c in word:
        if c in punct:
            return True
    return False

def replace_with_syn(word):
    lemma = lemmatizer.lemmatize(word, 'v')
    synonyms = []
    good_synonyms = []
    for syn in wordnet.synsets(lemma):
        for l in syn.lemmas():
            synonyms.append(l.name())
    if lemma in freq_lemmas:
        freq = True
    else:
        freq = False
    for synonym in synonyms:
        if synonym in freq_lemmas:
            freq_syn = True
        else:
            freq_syn = False
        if freq_syn ^ freq and not contain_punct(synonym):
            good_synonyms.append(synonym)
    return good_synonyms