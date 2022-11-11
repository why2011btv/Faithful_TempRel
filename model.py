import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from transformers import RobertaConfig, RobertaModel
from transformers import BigBirdConfig, BigBirdModel
from transformers import AutoConfig, AutoModel, AutoTokenizer
from all_loss_aug import transitivity_loss_H_, transitivity_loss_T_, cross_category_loss_, segment_loss_
from dpn_losses import DirichletKLLoss
import numpy as np
from opt_einsum import contract

'''
HiEve Stats
'''
HierPC_h = 1802.0
HierCP_h = 1846.0
HierCo_h = 758.0
HierNo_h = 63755.0 
HierTo_h = HierPC_h + HierCP_h + HierCo_h + HierNo_h # total number of event pairs
hier_weights_h = [0.25*HierTo_h/HierPC_h, 0.25*HierTo_h/HierCP_h, 0.25*HierTo_h/HierCo_h, 0.25*HierTo_h/HierNo_h]

'''
IC Stats
'''
HierPC_i = 2248.0 # before ignoring implicit events: 2257
HierCP_i = 2338.0 # 2354
HierCo_i = 2353.0 # 2358
HierNo_i = 81887.0 # 81857
HierTo_i = HierPC_i + HierCP_i + HierCo_i + HierNo_i # total number of event pairs
hier_weights_i = [0.25*HierTo_i/HierPC_i, 0.25*HierTo_i/HierCP_i, 0.25*HierTo_i/HierCo_i, 0.25*HierTo_i/HierNo_i]

temp_weights = [0.25*818.0/412.0, 0.25*818.0/263.0, 0.25*818.0/30.0, 0.25*818.0/113.0]

# transformers + MLP + Constraints
class transformers_mlp_cons(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.transformers_model = params['transformers_model']
        #self.model = AutoModel.from_pretrained(self.transformers_model)
        self.model = params['model']
        self.cuda = params['cuda']
        self.dataset = params['dataset']
        self.block_size = params['block_size']
        self.add_loss = params['add_loss']
        self.dpn = params['dpn']

        if self.dpn:
            self.target_concentration = 100.0
            self.id_criterion = DirichletKLLoss(target_concentration=self.target_concentration,
                                       concentration=1.0,
                                       reverse=True)
            self.ood_criterion = DirichletKLLoss(target_concentration=0.0,
                                        concentration=1.0,
                                        reverse=True)
            self.out_class = 3
        else:
            self.hier_class_weights_h = torch.FloatTensor(hier_weights_h).to(self.cuda)
            self.hier_class_weights_i = torch.FloatTensor(hier_weights_i).to(self.cuda)
            self.temp_class_weights = torch.FloatTensor(temp_weights).cuda()
            self.HiEve_anno_loss = nn.CrossEntropyLoss(weight=self.hier_class_weights_h)
            self.IC_anno_loss = nn.CrossEntropyLoss(weight=self.hier_class_weights_i)
            self.MATRES_anno_loss = nn.CrossEntropyLoss(weight=self.temp_class_weights)
            self.transitivity_loss_H = transitivity_loss_H_()
            self.transitivity_loss_T = transitivity_loss_T_()
            self.cross_category_loss = cross_category_loss_()
            self.out_class = 4
        
        self.emb_size = params['emb_size']
        self.fc = nn.Linear(2 * self.emb_size, self.emb_size)
        self.bilinear = nn.Linear(self.emb_size * self.block_size, self.out_class) # 3 if DPN; else 4
        self.fc1 = nn.Linear(3 * self.emb_size, self.emb_size)
        self.fc2 = nn.Linear(self.emb_size, self.out_class) # 3 if DPN; else 4
        
    def forward(self, input_ids, attention_mask, event_pos, event_pos_end, event_pair, labels):
        """ Encode with Transformer """
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        seq_output = output[0]
        attention = output[-1][-1]
        attention_dim = attention.size()[-1]
        seq_len = seq_output.size()[1]
        
        if attention_dim != seq_len:
            sequence_output = torch.zeros(seq_output.size()[0], attention_dim, seq_output.size()[2]).to(self.cuda)
            sequence_output[:, :seq_len, :] = seq_output
        else:
            sequence_output = seq_output
        #print(sequence_output.shape) #[batch_size, doc_len, 768]
        #print(attention.shape) # [batch_size, 12, doc_len, doc_len]
        
        """ Get representation for event pairs """
        batch_size = input_ids.size(0)
        e1_embs_batch = []
        e2_embs_batch = []
        atts_contract = []
        for i in range(batch_size):
            event_embs = []
            event_atts = []
            event_num_in_this_article = len(event_pos[i])
            for j in range(event_num_in_this_article):
                #e_emb = torch.mean(sequence_output[i, event_pos[i][j]:event_pos_end[i][j], :].unsqueeze(0), dim=1)
                e_emb = sequence_output[i, event_pos[i][j], :] # [768]
                event_embs.append(e_emb)
                e_att = attention[i, :, event_pos[i][j]] # [12, doc_len]
                event_atts.append(e_att)
                
            event_embs = torch.squeeze(torch.stack(event_embs, dim = 0))
            event_atts = torch.squeeze(torch.stack(event_atts, dim = 0))
            e1_embs = torch.index_select(event_embs, 0, torch.add(torch.tensor(event_pair[i]).to(event_embs.device)[:, 0], -1))
            e2_embs = torch.index_select(event_embs, 0, torch.add(torch.tensor(event_pair[i]).to(event_embs.device)[:, 1], -1))
            e1_embs_batch.append(e1_embs)
            e2_embs_batch.append(e2_embs)
            e1_atts = torch.index_select(event_atts, 0, torch.add(torch.tensor(event_pair[i]).to(event_embs.device)[:, 0], -1)) # torch.Size([780, 12, 428])
            e2_atts = torch.index_select(event_atts, 0, torch.add(torch.tensor(event_pair[i]).to(event_embs.device)[:, 1], -1))
            event_pair_att = (e1_atts * e2_atts).mean(1)
            event_pair_att = event_pair_att / (event_pair_att.sum(1, keepdim=True) + 1e-5) # torch.Size([780, 428])
            event_pair_contract = contract("ld,rl->rd", sequence_output[i], event_pair_att) # torch.Size([780, 768])
            atts_contract.append(event_pair_contract)
            
        e1_embs_batch = torch.cat(e1_embs_batch, dim = 0)
        e2_embs_batch = torch.cat(e2_embs_batch, dim = 0)
        #print("e1_embs_batch.shape:", e1_embs_batch.shape) # [pairs_num, 768]
        atts_contract = torch.cat(atts_contract, dim = 0)
        assert self.emb_size == e1_embs_batch.size(1)
        
        """ Calculating loss """    
        wenxuan = True
        if wenxuan:
            e1_representation = torch.tanh(self.fc(torch.cat([e1_embs_batch, atts_contract], dim = 1)))
            e2_representation = torch.tanh(self.fc(torch.cat([e2_embs_batch, atts_contract], dim = 1)))
            gb1 = e1_representation.view(-1, self.emb_size // self.block_size, self.block_size) # group bilinear
            gb2 = e2_representation.view(-1, self.emb_size // self.block_size, self.block_size) # group bilinear
            bl = (gb1.unsqueeze(3) * gb2.unsqueeze(2)).view(-1, self.emb_size * self.block_size) # ?
            logits = self.bilinear(bl)
        else:
            mul = torch.mul(e1_embs_batch, e2_embs_batch)
            logits = self.fc2(torch.tanh(self.fc1(torch.cat((e1_embs_batch, e2_embs_batch, mul), 1))))
        
        loss = 0.0
        labels = [torch.tensor(label) for label in labels]
        labels = torch.cat(labels, dim=0).to(logits).long()
        #print(labels.shape)
        #print(logits.shape)
        """
        if self.dataset == 'HiEve':
            loss += self.HiEve_anno_loss(logits, labels)
        elif self.dataset == 'IC':
            loss += self.IC_anno_loss(logits, labels)
        elif self.dataset == "MATRES":
            loss += self.MATRES_anno_loss(logits, labels)
        else:
            raise Exception('Dataset error!')
        """
        
        # Updated on Apr 13, 2022
        if self.dpn:
            for i in range(labels.shape[0]):
                if labels[i] != 3:
                    loss += temp_weights[int(labels[i])] * self.id_criterion(torch.unsqueeze(logits[i], 0), torch.unsqueeze(labels[i], 0)) / self.target_concentration
                else:
                    loss += temp_weights[int(labels[i])] * self.ood_criterion(torch.unsqueeze(logits[i], 0), None)
        else:
            loss += self.MATRES_anno_loss(logits, labels)

        if self.add_loss:
            print("adding loss...")
            start_index = 0
            for i in range(batch_size):
                event_num = len(event_pos[i])
                triples = []
                for a in range(0, event_num):
                    for b in range(a+1, event_num):
                        for c in range(b+1, event_num):
                            triples.append([a, b, c])
                alpha_logits = torch.index_select(logits[start_index:start_index+len(event_pair[i])], 0, torch.tensor(triples).to(event_embs.device)[:, 0])
                beta_logits = torch.index_select(logits[start_index:start_index+len(event_pair[i])], 0, torch.tensor(triples).to(event_embs.device)[:, 1])
                gamma_logits = torch.index_select(logits[start_index:start_index+len(event_pair[i])], 0, torch.tensor(triples).to(event_embs.device)[:, 2])
                if self.dataset in ['HiEve', 'IC']:
                    loss += self.add_loss * self.transitivity_loss_H(alpha_logits, beta_logits, gamma_logits).sum()
                else:
                    loss += self.add_loss * self.transitivity_loss_T(alpha_logits, beta_logits, gamma_logits).sum()
                start_index += len(event_pair[i])
            #print(start_index)
            #print(len(logits))
            #print(logits.shape)
            assert start_index == len(logits)
        return logits, loss
