import time
import numpy as np
from document_reader import *
import os
import os.path
from os import path
from os import listdir
from os.path import isfile, join
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from metric import metric, CM_metric
import json
from json import JSONEncoder
#import notify
#from notify_message import *
#from notify_smtp import *
from util import *

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
class exp:
    def __init__(self, cuda, model, epochs, learning_rate, train_dataloader, valid_dataloader, test_dataloader, dataset, best_PATH, load_model_path, dpn, model_name = None, relation_stats = None, lambdas = None):
        self.cuda = cuda
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dpn = dpn
        label_weights = []
        if relation_stats:
            for label in relation_stats.keys():
                label_weights.append(relation_stats[label])
            self.relation_stats = [w / sum(label_weights) for w in label_weights]
        if lambdas:
            self.lambda_1 = lambdas[0]
            self.lambda_2 = lambdas[1]
        if self.dpn == 1:
            self.out_class = 3
        else:
            self.out_class = 4
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        
        self.HiEve_best_F1 = -0.000001
        self.HiEve_best_prfs = []
        self.HiEve_best_PATH = best_PATH # to save model params here
        
        self.IC_best_F1 = -0.000001
        self.IC_best_prfs = []
        self.IC_best_PATH = best_PATH # to save model params here
        
        self.MATRES_best_F1 = -0.000001
        self.MATRES_best_cm = []
        self.MATRES_best_PATH = best_PATH # to save model params here
        
        self.best_epoch = 0
        self.load_model_path = load_model_path # load pretrained model parameters for testing, prediction, etc.
        self.model_name = model_name
        self.file = open("./rst_file/" + model_name + ".rst", "a")

    def train(self):
        total_t0 = time.time()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, amsgrad=True) # AMSGrad
        for epoch_i in range(0, self.epochs):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()
            self.model.train()
            self.total_train_loss = 0.0
            
            # https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html
            # batch accumulation parameter
            accum_iter = 1
            
            for step, batch in enumerate(self.train_dataloader):
                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader), elapsed))
                    
                logits, loss = self.model(batch[0].to(self.cuda), batch[1].to(self.cuda), batch[2], batch[3], batch[4], batch[5])
                
                #Don't update the model with loss_eo, since this is not the learning objective
                #logits_eo, loss_eo = self.model(batch[0].to(self.cuda), batch[6].to(self.cuda), batch[2], batch[3], batch[4], batch[5]) # Updated on May 17, 2022
                self.total_train_loss += loss.item()
                
                # normalize loss to account for batch accumulation
                loss = loss / accum_iter 
                
                # backward pass
                loss.backward()
                
                # weights update
                if ((step + 1) % accum_iter == 0) or (step + 1 == len(self.train_dataloader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)
            print("")
            print("  Total training loss: {0:.2f}".format(self.total_train_loss))
            print("  Training epoch took: {:}".format(training_time))
            flag, F_score = self.evaluate(self.dataset)
            if flag == 1:
                self.best_epoch = epoch_i + 1
        print("")
        print("======== Training complete! ========")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

        if self.dataset in ["HiEve"]:
            print("  HiEve best F1_PC_CP_avg: {0:.3f}".format(self.HiEve_best_F1))
            print("  HiEve best precision_recall_fscore_support:\n", self.HiEve_best_prfs)
            # Writing training results to file
            print("  Dev best:", file = self.file)
            print("  HiEve best F1_PC_CP_avg: {0:.3f}".format(self.HiEve_best_F1), file = self.file)
            print("  HiEve best precision_recall_fscore_support:", file = self.file)
            print(self.HiEve_best_prfs, file = self.file)
        if self.dataset in ["IC"]:
            print("  IC best F1_PC_CP_avg: {0:.3f}".format(self.IC_best_F1))
            print("  IC best precision_recall_fscore_support:\n", self.IC_best_prfs)
            # Writing training results to file
            print("  Dev best:", file = self.file)
            print("  IC best F1_PC_CP_avg: {0:.3f}".format(self.IC_best_F1), file = self.file)
            print("  IC best precision_recall_fscore_support:", file = self.file)
            print(self.IC_best_prfs, file = self.file)
            
        return self.HiEve_best_F1, self.IC_best_F1
            
    def evaluate(self, eval_data, test = False, predict = False, f1_metric = 'macro'):
        # ========================================
        #             Validation / Test
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        # Also applicable to test set.
        # Return 1 if the evaluation of this epoch achieves new best results,
        # else return 0.
        t0 = time.time()
            
        if test:
            if self.load_model_path:
                self.model = torch.load(self.load_model_path + self.model_name + ".pt")
            elif eval_data == "HiEve":
                self.model = torch.load(self.HiEve_best_PATH)
            elif eval_data == "IC":
                self.model = torch.load(self.IC_best_PATH)
            elif eval_data == "MATRES":
                self.model = torch.load(self.MATRES_best_PATH)
            else:
                print("NOT LOADING ANY MODEL...")
                
            self.model.to(self.cuda)
            print("")
            print("loaded " + eval_data + " best model:" + self.model_name + ".pt")
            #if predict == False:
                #print("(from epoch " + str(self.best_epoch) + " )")
            print("(from epoch " + str(self.best_epoch) + " )")
            print("Running Evaluation on " + eval_data + " Test Set...")
            dataloader = self.test_dataloader
        else:
            # Evaluation
            print("")
            print("Running Evaluation on Validation Set...")
            dataloader = self.valid_dataloader
            
        self.model.eval()
        
        y_pred = []
        y_gold = []
        if self.out_class == 3:
            y_logits = np.array([[0.0, 1.0, 2.0]])
        else:
            y_logits = np.array([[0.0, 1.0, 2.0, 3.0]])
        
        # Evaluate for one epoch.
        for batch in dataloader:
            with torch.no_grad():
                logits, loss = self.model(batch[0].to(self.cuda), batch[1].to(self.cuda), batch[2], batch[3], batch[4], batch[5])
                logits_eo, loss_eo = self.model(batch[0].to(self.cuda), batch[6].to(self.cuda), batch[2], batch[3], batch[4], batch[5]) # Updated on May 17, 2022
                logits_xb, loss_xb = self.model(batch[0].to(self.cuda), batch[7].to(self.cuda), batch[2], batch[3], batch[4], batch[5]) # Updated on Jun 14, 2022
                logits = nn.Softmax(dim=1)(logits) - torch.tensor(self.lambda_1) * nn.Softmax(dim=1)(logits_eo) - torch.tensor(self.lambda_2) * nn.Softmax(dim=1)(logits_xb) # Updated on Jun 14, 2022
                
            # Move logits and labels to CPU
            y_predict = torch.max(logits[:, 0:self.out_class], 1).indices.cpu().numpy()
            y_pred.extend(y_predict)
            
            labels = []
            for batch_label in batch[5]:
                for label in batch_label:
                    labels.append(label)
            y_gold.extend(labels)
            
            y_logits = np.append(y_logits, logits[:, 0:self.out_class].cpu().numpy(), 0) # for prediction result output # 3 if DPN; else 4
            
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print("Eval took: {:}".format(validation_time))
        
        # Output prediction results.
        if predict:
            if predict[-4:] == "json":
                with open(predict, 'w') as outfile:
                    numpyData = {"labels": "0 -- Parent-Child or Before; 1 -- Child-Parent or After; 2 -- Coref or Simultaneous; 3 -- NoRel or Vague", "array": y_logits}
                    json.dump(numpyData, outfile, cls=NumpyArrayEncoder)
                #try:
                #    msg = message(subject=eval_data + " Prediction Notice",
                #                  text=self.dataset + "/" + self.model_name + " Predicted " + str(y_logits.shape[0] - 1) + " instances. (Current Path: " + os.getcwd() + ")")
                #    send(msg)  # and send it
                #except:
                #    print("Send failed.")
                #return 0
            else:
                with open(predict + "gold", 'w') as outfile:
                    for i in y_gold:
                        print(i, file = outfile)
                with open(predict + "pred", 'w') as outfile:
                    for i in y_pred:
                        print(i, file = outfile)   
        
        # Calculate the performance.
        
        if eval_data == "MATRES":
            try:  
                if self.dpn:
                    tri_gold = []
                    tri_pred = []
                    for i, label in enumerate(y_gold):
                        if label != 3:
                            tri_gold.append(label)
                            tri_pred.append(y_pred[i])
                    macro_f1 = f1_score(tri_gold, tri_pred, average='macro')
                    micro_f1 = f1_score(tri_gold, tri_pred, average='micro')
                    print("  macro F1: {0:.3f}".format(macro_f1))
                    print("  micro F1: {0:.3f}".format(micro_f1))
                    CM = confusion_matrix(tri_gold, tri_pred)
                    print(CM)
                else:
                    Acc, P, R, F1, CM = metric(y_gold, y_pred)
                    print("  P: {0:.3f}".format(P))
                    print("  R: {0:.3f}".format(R))
                    print("  F1: {0:.3f}".format(F1))
                    macro_f1 = f1_score(y_gold, y_pred, average='macro')
                    micro_f1 = f1_score(y_gold, y_pred, average='micro')
                    print("  macro f-score: {0:.3f}".format(macro_f1))
                    print("  micro f-score: {0:.3f}".format(micro_f1))
                    print(CM)
                if test:
                    tri_gold = []
                    tri_pred = []
                    prob = y_logits[1:]
                    for i, label in enumerate(y_gold):
                        if label != 3:
                            tri_prob = prob[i][0:3]
                            tri_gold.append(label)
                            tri_pred.append(np.argmax(tri_prob))
                            
                    macro_f1 = f1_score(tri_gold, tri_pred, average='macro')
                    micro_f1 = f1_score(tri_gold, tri_pred, average='micro')
                    print("  macro F1: {0:.3f}".format(macro_f1))
                    print("  micro F1: {0:.3f}".format(micro_f1))
                    
                    F1 = f1_score(tri_gold, tri_pred, average=f1_metric)
                    print("Test result:", file = self.file)
                    print("  "+f1_metric+" F1: {0:.3f}".format(F1), file = self.file)
                    #try:
                    #    msg = message(subject=eval_data + " Test Notice",
                    #          text = self.dataset + "/" + self.model_name + " Test results:\n" + "  F1: {0:.3f}".format(F1) + " (Current Path: " + os.getcwd() + ")")
                    #    send(msg)  # and send it
                    #except:
                    #    print("Send failed.")
                    return 2, F1
                if not test:
                    if F1 > self.MATRES_best_F1 or path.exists(self.MATRES_best_PATH) == False:
                        self.MATRES_best_F1 = F1
                        self.MATRES_best_cm = CM
                        ### save model parameters to .pt file ###
                        torch.save(self.model, self.MATRES_best_PATH)
                        return 1, F1
                    else:
                        return 0, F1
            except:
                print("No classification_report for this epoch of evaluation (Recall and F-score are ill-defined and being set to 0.0 due to no true samples).")
                
        '''        
        if eval_data in ["HiEve", "IC"]:
            try:
                # Report the final accuracy for this validation run.
                cr = classification_report(y_gold, y_pred, output_dict = True)
                rst = classification_report(y_gold, y_pred)
                F1_PC = cr['0']['f1-score']
                F1_CP = cr['1']['f1-score']
                F1_coref = cr['2']['f1-score']
                F1_NoRel = cr['3']['f1-score']
                F1 = (F1_PC + F1_CP) / 2.0
                print(rst)
                print("  F1_PC_CP_avg: {0:.3f}".format(F1))

                if test:
                    print("  Test rst:", file = self.file)
                    print(rst, file = self.file)
                    print("  F1_PC_CP_avg: {0:.3f}".format(F1), file = self.file)
                    msg = message(subject = eval_data + " Test Notice", text = self.dataset + "/" + self.model_name + " Test results:\n" + "  F1_PC_CP_avg: {0:.3f}".format(F1) + " (Current Path: " + os.getcwd() + ")")
                    send(msg)  # and send it

                if not test:
                    msg = message(subject = eval_data + " Validation Notice", text = self.dataset + "/" + self.model_name + " Validation results:\n" + "  F1_PC_CP_avg: {0:.3f}".format(F1) + " (Current Path: " + os.getcwd() + ")")
                    send(msg)  # and send it
                    if eval_data == "HiEve":
                        if F1 > self.HiEve_best_F1 or path.exists(self.HiEve_best_PATH) == False:
                            self.HiEve_best_F1 = F1
                            self.HiEve_best_prfs = rst
                            torch.save(self.model, self.HiEve_best_PATH)
                            return 1, F1
                    else:
                        if F1 > self.IC_best_F1 or path.exists(self.IC_best_PATH) == False:
                            self.IC_best_F1 = F1
                            self.IC_best_prfs = rst
                            torch.save(self.model, self.IC_best_PATH)
                            return 1, F1
            except:
                print("No classification_report for this epoch of evaluation (Recall and F-score are ill-defined and being set to 0.0 due to no true samples), or send failed.")
        '''
        return 0, F1
    


