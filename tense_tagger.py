#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Usage: python black_task.py TEXT_PATH
#
# Parse the text file specified in the TEXT_PATH,
# produce tense labels for each verb group, count the labels and elisions.
#
# The script also contains base functions used tense classification in the web-app.


import re

import nltk
from nltk import WordPunctTokenizer
from anytree.search import findall_by_attr
from anytree import Node


class TreeModel:
    root = Node("root")
    # root children
    vbp = Node("vbp", parent=root, tense="pressimp")
    vbz = Node("vbz", parent=root, tense="pressimp")
    vbd = Node("vbd", parent=root, tense="pastsimp")
    tobePres = Node("tobepres", parent=root, tense="pressimp")
    tobePast = Node("tobepast", parent=root, tense="pastsimp")
    md = Node("md", parent=root, tense=None)

    # vbp children
    vbp_vb = Node("vb", parent=vbp, tense="pressimp")
    vbp_vbn = Node("vbn", parent=vbp, tense="presperfsimp")
    vbp_vbn_vbn = Node("vbn", parent=vbp_vbn, tense="presperfsimppass")
    vbp_vbn_vbg = Node("vbg", parent=vbp_vbn, tense="presperfcont")
    vbp_vbn_vbg_vbn = Node("vbn", parent=vbp_vbn_vbg, tense="presperfcontpass")

    # vbz children
    vbz_vb = Node("vb", parent=vbz, tense="pressimp")
    vbz_vbn = Node("vbn", parent=vbz, tense="presperfsimp")
    vbz_vbn_vbn = Node("vbn", parent=vbz_vbn, tense="presperfsimppass")
    vbz_vbn_vbg = Node("vbg", parent=vbz_vbn, tense="presperfcont")
    vbz_vbn_vbg_vbn = Node("vbn", parent=vbz_vbn_vbg, tense="presperfcontpass")

    # vbd children
    vbd_vb = Node("vb", parent=vbd, tense="pastsimp")
    vbd_vbn = Node("vbn", parent=vbd, tense="pastperfsimp")
    vbd_vbn_vbn = Node("vbn", parent=vbd_vbn, tense="pastperfsimppass")
    vbd_vbn_vbg = Node("vbg", parent=vbd_vbn, tense="pastperfcont")
    vbd_vbn_vbg_vbn = Node("vbn", parent=vbd_vbn_vbg, tense="pastperfcontpass")

    # tobePres children
    tobePres_vbn = Node("vbn", parent=tobePres, tense="pressimppass")
    tobePres_vbg = Node("vbg", parent=tobePres, tense="prescont")
    tobePres_vbg_vbn = Node("vbn", parent=tobePres_vbg, tense="prescontpass")

    # tobePast children
    tobePast_vbn = Node("vbn", parent=tobePast, tense="pastsimppass")
    tobePast_vbg = Node("vbg", parent=tobePast, tense="pastcont")
    tobePast_vbg_vbn = Node("vbn", parent=tobePast_vbg, tense="pastcontpass")

    # md children
    md_vb = Node("vb", parent=md, tense="futusimp")
    md_vb_vbn = Node("vbn", parent=md_vb, tense="futuperfsimp")
    md_vb_vbn_vbn = Node("vbn", parent=md_vb_vbn, tense="futuperfsimppass")
    md_vb_vbn_vbg = Node("vbg", parent=md_vb_vbn, tense="futuperfcont")
    md_vb_vbn_vbg_vbn = Node("vbn", parent=md_vb_vbn_vbg, tense="futuperfcontpass")

    md_tobePres = Node("tobepres", parent=md, tense="futusimp")
    md_tobePres_vbg = Node("vbg", parent=md_tobePres, tense="futucont")
    md_tobePres_vbg_vbn = Node("vbn", parent=md_tobePres_vbg, tense="futucontpass")
    md_tobePres_vbn = Node("vbn", parent=md_tobePres, tense="futusimppass")


def check_to_be(token):
    tobe_pres = ['is', 'are', 'am']
    tobe_past = ['was', 'were']
    temp = list(token)
    if temp[0].lower() in tobe_pres:
        temp[1] = "tobepres"
    if temp[0].lower() in tobe_past:
        temp[1] = "tobepast"
    return tuple(temp)


def get_verbs(tokens, tokens_idx, verbose=False):
    verbs_list = ['VBP', 'VBN', 'VBZ', 'VBD', 'VBG', 'MD', 'VB']

    def equal_vb_pos(toks, idx):
        # previous verb pos tag must equal the next one for elision condition to pass
        if idx < 1:
            return False
        last_verb_pos = toks[idx - 1][1]
        next_verb_pos = ""
        for tok in toks[idx + 1:]:
            if tok[1] in verbs_list:
                next_verb_pos = tok[1]
                break
        return last_verb_pos == next_verb_pos

    verb_groups = []
    verbs = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if verbose:
            print("token in get_verbs: ", token)
        if token[1] in verbs_list:
            token += ([tokens_idx[i][0], tokens_idx[i][1]],)
            if verbose:
                print("in if, token: ", token)
            verbs.append(token)
        elif (token[1] == ',' or token[1] == 'CC') and equal_vb_pos(tokens, i):  # elisions - conjunctions between verbs
            pass
        elif re.match("RB*", token[1]):  # adjective between verbs
            pass
        elif re.match("NN*", token[1]) or re.match("PRP*", token[1]):  # noun or personal pronoun between verbs
            pass
        else:
            if len(verbs) != 0:
                verb_groups.append(verbs)
            verbs = []
        i += 1
    if len(verbs) != 0:
        verb_groups.append(verbs)
    for verb_group in verb_groups:
        for i in range(len(verb_group)):
            verb_group[i] = check_to_be(verb_group[i])
    return verb_groups


def get_tense(tokens, verbose=False):
    starting_node = TreeModel.root
    mark = 0
    for i in range(len(tokens)):
        token = tokens[i]
        if verbose:
            print("tokens: ", tokens)
            print("token: ", token)
        found = findall_by_attr(starting_node, token[1].lower(), maxlevel=2)
        if len(found) != 0:
            starting_node = found[0]
            if i == len(tokens) - 1:
                for j in range(mark, i + 1):
                    tense = starting_node.tense
                    depth = starting_node.depth
                    tokens[j] += (tense, depth,)  # ',' to make tuple
        else:
            if starting_node != TreeModel.root:
                for j in range(mark, i):
                    tense = starting_node.tense
                    depth = starting_node.depth
                    tokens[j] += (tense, depth,)  # ',' to make tuple
                starting_node = TreeModel.root
                i -= 1
            mark = i
    return tokens


def manual_pos_correction(tokens):
    corrections = {
        "Did": ('Did', 'VBD'),
        "Are": ('Are', 'VBP'),
    }
    for idx, token in enumerate(tokens):
        if token[0] in corrections:
            tokens[idx] = corrections[token[0]]

    return tokens


def get_tense_verb_groups(text, verbose=False):
    tokens = WordPunctTokenizer().tokenize(text)
    if verbose:
        print("*************", len(tokens))
    tokens_idx = list(WordPunctTokenizer().span_tokenize(text))
    tokens = nltk.pos_tag(tokens)
    tokens = manual_pos_correction(tokens)
    if verbose:
        print("================", len(tokens), len(tokens_idx))
        print("tokens: ", tokens)
    verb_groups = get_verbs(tokens, tokens_idx)
    if verbose:
        print("verb_groups: ", verb_groups)
    tense_verb_groups = []
    for verb_group in verb_groups:
        tense_verb_group = get_tense(verb_group, verbose=verbose)
        tense_verb_groups.append(tense_verb_group)
    return tense_verb_groups


def print_parsed_text(text):
    tense_vb_groups = get_tense_verb_groups(text, verbose=False)
    # initialize counts for all possible labels
    all_tenses = {
        'futuperfsimp', 'futucont', 'futucontpass', 'futuperfcont',
        'futuperfcontpass', 'futuperfsimppass', 'futusimp', 'futusimppass',
        'pastcont', 'pastcontpass', 'pastperfcont', 'pastperfcontpass',
        'pastperfsimp', 'pastperfsimppass', 'pastsimp', 'pastsimppass',
        'prescont', 'prescontpass', 'presperfcont', 'presperfcontpass',
        'presperfsimp', 'presperfsimppass', 'pressimp', 'pressimppass'
    }
    labels_elisions_counts = {tense: {"labels": 0, "elisions": 0} for tense in all_tenses}

    out_txt = text
    span_correction = 0
    tense_list = []
    for vb_group in tense_vb_groups:
        tree_depth = 0
        tense_vb_count = 0
        last_token_span = []
        vb_tense = ""
        for verb in vb_group:
            if len(verb) == 5:  # verb with a successfully classified tense
                last_token_span, vb_tense, tree_depth = verb[2:]
                tense_list.append(verb)
                tense_vb_count += 1
        """
        if tense_vb_count != 0:
            labels_elisions_counts[vb_tense]["labels"] += 1
            labels_elisions_counts[vb_tense]["elisions"] += tense_vb_count - tree_depth
            insertion_index = last_token_span[1] + span_correction
            insertion = '<{}>'.format(vb_tense)
            out_txt = out_txt[:insertion_index] + insertion + out_txt[insertion_index:]
            span_correction += len('<{}>'.format(vb_tense))
        """
    #print(format_text_output(all_tenses, labels_elisions_counts, out_txt))
    return tense_list


def format_text_output(all_tenses, labels_elisions_nbs, output_text):
    print("\n" + "-" * 79 + "\n" + output_text)
    label_margin = max([len(tense) for tense in all_tenses])
    counts_txt = "-" * 79
    for tense, counts in sorted(labels_elisions_nbs.items()):
        counts_txt += "\n" + tense + ": " + " " * (label_margin - len(tense))
        for count_name, count in counts.items():
            counts_txt += "{} {}; ".format(count, count_name)
    return counts_txt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Tool to classify tenses and count the number of tense labels & elisions.'
    )
    parser.add_argument('text_path', help='A path to the text file to be processed.')
    args = parser.parse_args()

    with open(args.text_path, 'r') as f:
        txt = f.read()

    # txt = """
    # In Gatlin, South Carolina, teenager Ethan Wate awakens from a recurring dream of a girl he does not know.
    # In voice-over narration, he describes his enjoyment of reading banned books, his despair of his small-town
    # existence, and his dreams of leaving Gatlin for college. Arriving for his first day of junior year, Ethan notices
    # newcomer Lena Duchannes, who resembles the girl he has been dreaming about. The other students do not take kindly
    # to her and spread gossip regarding Lena's reclusive uncle, Macon Ravenwood, and suggest that her family includes
    # devil worshippers. Overhearing these whispers, Lena tenses. On a drive home, Ethan nearly runs over Lena,
    # whose car has broken down. He gives her a ride home, and the two bond over their shared love of poetry and having
    # both lost their mothers.
    # """
    # txt = "The item is packed, is checked and is delivered."
    # txt = "The item is packed, checked and delivered."
    # txt = "It had been working and functioning properly."

    # txt = "Did anything happen? Are you okay? When do you want to start? Is he OK?"
    # txt = "He does not know!"

    print(print_parsed_text(txt))
