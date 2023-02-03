import numpy as np
import json
from os import listdir
from os.path import isfile, join
import networkx as nx

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def editDistance(str1, str2, m, n):
 
    # If first string is empty, the only option is to
    # insert all characters of second string into first
    if m == 0:
        return n
 
    # If second string is empty, the only option is to
    # remove all characters of first string
    if n == 0:
        return m
 
    # If last characters of two strings are same, nothing
    # much to do. Ignore last characters and get count for
    # remaining strings.
    if str1[m-1] == str2[n-1]:
        return editDistance(str1, str2, m-1, n-1)
 
    # If last characters are not same, consider all three
    # operations on last character of first string, recursively
    # compute minimum cost for all three operations and take
    # minimum of three values.
    return 1 + min(editDistance(str1, str2, m, n-1),    # Insert
                   editDistance(str1, str2, m-1, n),    # Remove
                   editDistance(str1, str2, m-1, n-1)    # Replace
                   )

class temporal_graph:
    def __init__(self, edgelist, confidence_list):  
        self.edges = {}
        self.confidence = {}
        for edge_id, edge in enumerate(edgelist):
            self.edges[edge_id] = {'edge': edge, 'confidence': confidence_list[edge_id]}
            self.confidence[edge_id] = confidence_list[edge_id]            
        self.g = nx.DiGraph([self.edges[edge_id]['edge'] for edge_id in self.edges.keys()])
        #self.g = nx.transitive_reduction(DG)
        
    def remove_edge(self):
        sorted_conf = list(sorted(self.confidence.items(), key=lambda x: x[1]))
        edge_id_to_remove = sorted_conf[0][0]
        self.confidence.pop(edge_id_to_remove)
        #self.edges.pop(edge_id_to_remove)
        return edge_id_to_remove
        
    def construct_timeline(self):
        while list(nx.simple_cycles(self.g)) != []:
            edge_id_to_remove = self.remove_edge()
            self.g.remove_edge(self.edges[edge_id_to_remove]['edge'][0], self.edges[edge_id_to_remove]['edge'][1])
            #print(self.g.edges())
            #g = nx.DiGraph([self.edges[edge_id]['edge'] for edge_id in self.edges.keys()])
        #edge_list = [(self.edges[edge_id]['edge'][0], self.edges[edge_id]['edge'][0], {'conf': self.confidence[edge_id]}) for edge_id in self.edges.keys()]
        #final_g = nx.DiGraph(edge_list)
        TR = nx.transitive_reduction(self.g)
        return list(nx.all_topological_sorts(self.g))#, list(TR.edges)
    
    def longest_path(self):
        while list(nx.simple_cycles(self.g)) != []:
            edge_id_to_remove = self.remove_edge()
            self.g.remove_edge(self.edges[edge_id_to_remove]['edge'][0], self.edges[edge_id_to_remove]['edge'][1])
            #print(self.g.edges())
        return nx.dag_longest_path(self.g)
    
    def remove_unconfident_edge(graph):
        edge_to_w = {}
        for edge in graph.edges:
            edge_to_w[edge] = DG[edge[0]][edge[1]]['confidence']
        edge_to_remove = min(edge_to_w, key=edge_to_w.get)
        return edge_to_remove
    
    def longest_path_in_graph(self, graph):
        while list(nx.simple_cycles(graph)) != []:
            edge_to_remove = self.remove_unconfident_edge(graph)
            graph.remove_edge(edge_to_remove[0], edge_to_remove[1])
            #print(self.g.edges())
        return nx.dag_longest_path(graph)
    
    def longest_path_in_each_cc(self):
        S = [self.g.subgraph(c).copy() for c in nx.weakly_connected_components(self.g)]
        timelines = []
        for cc in S:
            timelines.append(self.longest_path_in_graph(cc))
        return timelines
"""            
import json
with open("/shared/why16gzl/Repositories/LEC_OnePass/tdd_labels.json", 'r') as f:
    tdd_labels = json.load(f)
def tdd_perf(json_file, long = True):
    with open("/shared/why16gzl/Repositories/LEC_OnePass/prediction/"+json_file, 'r') as f:
        pred = json.load(f)
        logits = pred['array'][1:]
        y_conf = np.max(softmax(np.array(logits)), axis = 1)
        #print(len(y_conf))
        y_pred = np.argmax(logits, axis = 1)
        #print(len(y_pred))

    pointer = 0
    correct = 0
    tdd = {}
    edit_distance = 0
    for tl_id in tdd_labels.keys():
        tdd[tl_id] = {'event_pairs': [], 'conf': []}
        for pair_id, e_pair in enumerate(tdd_labels[tl_id]['event_pairs']):
            if y_pred[pointer + pair_id] == 0:
                tdd[tl_id]['conf'].append(y_conf[pointer + pair_id])
                tdd[tl_id]['event_pairs'].append([e_pair[0], e_pair[1]])
            else:
                #print("before reverse", tdd_labels[tl_id]['event_pairs'][pair_id])
                tdd[tl_id]['conf'].append(y_conf[pointer + pair_id])
                tdd[tl_id]['event_pairs'].append([e_pair[1], e_pair[0]])

        pointer += len(tdd[tl_id]['event_pairs'])
        tg = temporal_graph(tdd[tl_id]['event_pairs'], tdd[tl_id]['conf'])
        if long:
            pred_tl = tg.longest_path()
        else:
            pred_tl = tg.construct_timeline()

        if tl_id == '264':
            print(tl_id)
            print(tdd[tl_id])
            print(tdd_labels[tl_id]['timeline'], pred_tl)
            #print(tdd[tl_id]['event_pairs'][removed_edge], tdd[tl_id]['conf'][removed_edge])
            print(tdd_labels[tl_id]['timeline'] == pred_tl)
            print()
            
        edit_distance += editDistance(tdd_labels[tl_id]['timeline'], pred_tl, len(tdd_labels[tl_id]['timeline']), len(pred_tl))

        if tdd_labels[tl_id]['timeline'] == pred_tl:
            correct += 1
        #else:
        #    print("tg edges:", tg.g.edges, "gold tl:", tdd_labels[tl_id]['timeline'], "pred tl:", pred_tl)
    
    print(round(edit_distance / len(tdd_labels), 2))
    print(round(100 * correct / len(tdd_labels), 2))
    print()
#tdd_perf(json_file, False)
#tdd_perf(json_file, True)
"""

def check_feasibility(timeline, edges):
    num_node = len(timeline)
    for i in range(num_node-1):
        if [timeline[i], timeline[i+1]] not in edges:
            return False
    return True
           

def tl_construction(logits, art_split, long = True, softmax = True, predict_with_abstention = []):
    if softmax:
        y_conf = np.max(softmax(logits), axis = 1)
    else:
        y_conf = np.max(logits, axis = 1)
    #print(len(y_conf))
    y_pred = np.argmax(logits, axis = 1)
    #print(len(y_pred))

    i = -1
    tdd = {}
    edit_distance = 0
    last_end = 0
    #for tl_id, pairs in enumerate(art_split):
    for tl_id in art_split.keys():
        pairs = art_split[tl_id]

        tdd[tl_id] = {'event_pairs': [], 'conf': [], 
                      'event_pairs_pwa': [], 'conf_pwa': []}
        for pair in pairs:
            i += 1
            if y_pred[i] == 0:
                tdd[tl_id]['conf'].append(y_conf[i])
                tdd[tl_id]['event_pairs'].append(pair)
            if y_pred[i] == 1:
                tdd[tl_id]['conf'].append(y_conf[i])
                tdd[tl_id]['event_pairs'].append([pair[1], pair[0]])
            if predict_with_abstention != []:
                if y_pred[i] == 0 and predict_with_abstention[i] == 1:
                    tdd[tl_id]['conf_pwa'].append(y_conf[i])
                    tdd[tl_id]['event_pairs_pwa'].append(pair)
                if y_pred[i] == 1 and predict_with_abstention[i] == 1:
                    tdd[tl_id]['conf_pwa'].append(y_conf[i])
                    tdd[tl_id]['event_pairs_pwa'].append([pair[1], pair[0]])
                
        #tg = temporal_graph(tdd[tl_id]['event_pairs'], tdd[tl_id]['conf'])
        tg_pwa = temporal_graph(tdd[tl_id]['event_pairs_pwa'], tdd[tl_id]['conf_pwa'])

        #tdd[tl_id]['timeline_topo_sort'] = tg.construct_timeline()
        #tdd[tl_id]['timeline_longest_path'] = tg.longest_path()
        #tdd[tl_id]['timeline_longest_path_pwa'] = tg_pwa.longest_path_in_each_cc()
        #tdd[tl_id]['timeline_topo_sort_pwa'] = tg_pwa.construct_timeline()
        
        timelines = tg_pwa.construct_timeline()
        tdd[tl_id]['timeline_topo_sort_pwa'] = []
        for timeline in timelines:
            if check_feasibility(timeline, tdd[tl_id]['event_pairs']):
                tdd[tl_id]['timeline_topo_sort_pwa'].append(timeline)

    return tdd

