from __future__ import print_function
import csv
import logging
import argparse
from math import exp
import math
from copy import copy
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
from torchtext import data
import nltk
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import datapath
import time
from lm import NGramLangModel
from util import *
import spacy
import wmd
import re
import torch.nn.functional as F
#nlp = spacy.load('en', create_pipeline=wmd.WMD.create_spacy_pipeline)
nlp = spacy.load('en_core_web_sm', create_pipeline=wmd.WMD.create_spacy_pipeline)
NGRAM = 3
TAU = 0.7
N_NEIGHBOR = 15
N_REPLACE = 5

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('word_delta', help= 'percentage of allowed word paraphasing')
    parser.add_argument('model', help='model: either CNN or LSTM')
    parser.add_argument('train_path', help='Path to training data')
    parser.add_argument('test_path', help='Path to testing data')
    parser.add_argument('output_path', help='Path to output changed test data')
    parser.add_argument('--embedding_path', action='store', dest='embedding_path',
        help='Path to pre-trained embedding data')
    parser.add_argument('--model_path', action='store', dest='model_path',
        help='Path to pre-trained classifier model')
    parser.add_argument('max_size', help='max amount of transformations to be processed by each iteration')
    parser.add_argument('--first_label', help='The name of the first label that the model sees in the \
                         training data. The model will automatically set it to be the positive label. \
                         For instance, in the fake news dataset, the first label is FAKE.')
    return parser.parse_args()
#default="FAKE"


class CNN(nn.Module):
    def __init__(self, sentence_len=200, kernel_sizes=[3,4,5], num_filters=100, embedding_dim=300, pretrained_embeddings=None):
        super(CNN, self).__init__()
        self.sentence_len=sentence_len
        use_cuda = torch.cuda.is_available()
        self.kernel_sizes = kernel_sizes
        vocab_size=len(pretrained_embeddings)
        print(vocab_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = False #mode=="nonstatic"
        if use_cuda:
            self.embedding = self.embedding.cuda()
        conv_blocks = []
        for kernel_size in kernel_sizes:
            # maxpool kernel_size must <= sentence_len - kernel_size+1, otherwise, it could output empty
            maxpool_kernel_size = sentence_len - kernel_size +1
            conv1d = nn.Conv1d(in_channels = 1, out_channels = num_filters, kernel_size = kernel_size*embedding_dim, stride = embedding_dim)

            component = nn.Sequential(
                conv1d,
                nn.ReLU(),
                nn.MaxPool1d(kernel_size = maxpool_kernel_size)
            )
            if use_cuda:
                component = component.cuda()

            conv_blocks.append(component)
        self.conv_blocks = nn.ModuleList(conv_blocks)   # ModuleList is needed for registering parameters in conv_blocks
        self.fc = nn.Linear(num_filters*len(kernel_sizes), 2)

    def forward(self, x):       # x: (batch, sentence_len)
        x = self.embedding(x)   # embedded x: (batch, sentence_len, embedding_dim)
        #    input:  (batch, in_channel=embedding_dim, in_length=sentence_len),
        #    output: (batch, out_channel=num_filters, out_length=sentence_len-...)
        x = x.view(x.size(0), 1, -1)  # needs to convert x to (batch, 1, sentence_len*embedding_dim)
        x_list= [conv_block(x) for conv_block in self.conv_blocks]
        out = torch.cat(x_list, 2)
        out = out.view(out.size(0), -1)
        feature_extracted = out
        return F.softmax(self.fc(out), dim=1), feature_extracted


class Attacker(object):
    ''' main part of the attack model '''
    def __init__(self, X, opt):
        self.opt=opt
        self.suffix='wordonly-'+str(opt.word_delta)
        self.DELTA_W=int(opt.word_delta)*0.1
        self.TAU_2=2
        self.TAU_wmd_s = 0.75
        self.TAU_wmd_w=0.75
        # want do sentence level paraphrase first
        X=[doc.split() for doc in X]
        logging.info("Initializing language model...")
        print("Initializing language model...")
        self.lm = NGramLangModel(X, NGRAM)
        logging.info("Initializing word vectors...")
        print("Initializing word vectors...")
        #print(opt.embedding_path)
        #self.w2v = KeyedVectors.load_word2vec_format(datapath('home/vlad-ubuntu/classifiers/adversarial_text/paragram_300_sl999/paragram_300_sl999/paragram_300_sl999.txt'), binary=False)
        #self.w2v = KeyedVectors.load_word2vec_format(opt.embedding_path, binary=False)
        self.w2v = KeyedVectors.load_word2vec_format(opt.embedding_path, binary=False, unicode_errors='ignore')


        logging.info("Loading pre-trained classifier...")
        print("Loading pre-trained classifier...")
        self.model = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
        if torch.cuda.is_available():
            self.model.cuda()
        logging.info("Initializing vocabularies...")
        print("Initializing vocabularies...")
        self.src_vocab, self.label_vocab = self.load_vocab(opt.train_path)
        # to compute the gradient, we need to set up the optimizer first
        self.criterion = nn.CrossEntropyLoss()

    def word_paraphrase(self, words, poses, list_neighbors, y):
            candidates = [words]
            j=1
            if self.opt.model=='LSTM':
                max_size=int(self.opt.max_size)//len(words)
            else:
                max_size=int(self.opt.max_size)//self.model.sentence_len
            for pos in poses:
                    closest_neighbors=list_neighbors[pos]
                    if not closest_neighbors:
                        j+=1
                        continue
                    current_candidates= copy(candidates)
                    for repl in closest_neighbors:
                        for c in candidates:
                            if len(current_candidates)>max_size:
                                break
                            corrupted = copy(c)
                            corrupted[pos] = repl
                            current_candidates.append(corrupted)
                    candidates=copy(current_candidates)
                    if len(candidates)>max_size:
                        break
                    j+=1
            if candidates:
                if self.opt.model=='LSTM':
                    candidate_var = text_to_var(candidates, self.src_vocab)
                    pred_probs = self.model(candidate_var)
                    log_pred_prob, best_candidate_id = pred_probs[:, 1-y].max(dim=0)
                    #new_words = candidates[best_candidate_id.data[0]]
                    #pred_prob = exp(log_pred_prob.data[0])
                    new_words = candidates[best_candidate_id.data]
                    pred_prob = exp(log_pred_prob.data)
                elif self.opt.model=='CNN':
                    candidate_var = self.text_to_var_CNN(candidates, self.src_vocab)
                    pred_probs,_ = self.model(candidate_var)
                    log_pred_prob, best_candidate_id = pred_probs[:, 1-y].max(dim=0)
                    #new_words = candidates[best_candidate_id.data[0]]
                    #pred_prob = log_pred_prob.data[0]
                    new_words = candidates[best_candidate_id.data]
                    pred_prob = log_pred_prob.data
                    #VladsOutput.write("\n THIS IS PRED PROB " + str(pred_prob) + "\n")
            else:
                print('empty candidates!')
            return new_words, pred_prob, j

    def hidden(self, hidden_dim):
        if torch.cuda.is_available():
            h0=Variable(torch.zeros(1,1,hidden_dim).cuda())
            c0=Variable(torch.zeros(1,1,hidden_dim).cuda())
        else:
            h0=Variable(torch.zeros(1,1,hidden_dim))
            c0=Variable(torch.zeros(1,1,hidden_dim))
        return (h0,c0)

    def forward_lstm(self, embed,model):  #copying the structure of LSTMClassifer, just omitting the first embedding layer
        lstm_out, hidden0= model.rnn(embed, self.hidden(512))
        y=model.linear(lstm_out[-1])
        return y
    def forward_cnn(self,embed,model):
        x_list= [conv_block(embed) for conv_block in model.conv_blocks]
        out = torch.cat(x_list, 2)
        out = out.view(out.size(0), -1)
        return F.softmax(model.fc(out), dim=1)
    def text_to_var_CNN(self, docs, vocab):
        tensor = []
        max_len = self.model.sentence_len 
        for doc in docs:
            vec = []
            for tok in doc:
                vec.append(vocab.stoi[tok])
            if len(doc) < max_len:
                vec += [0]*(max_len-len(doc))   
            else:
                vec=vec[:max_len]
            tensor.append(vec)
        var = Variable(torch.LongTensor(tensor))
        if torch.cuda.is_available():
            var = var.cuda()
        return var

    
    def sentence_paraphrase(self, y, sentences, changed_pos, list_closest_neighbors):
            candidates = []
            responding_pos = [] # the index of the changed sentence
            for i, sentence in enumerate(sentences):
                if i in changed_pos:
                    continue
                j=0
                for p in list_closest_neighbors[i]:
                    new_sentence=copy(sentences)
                    new_sentence[i]=p
                    new_sentence=(" ".join(new_sentence)).split()
                    candidates.append(new_sentence)
                    responding_pos.append((i,j))
                    j+=1

            if candidates:
                m=len(candidates)
                if self.opt.model=='LSTM':
                    n=max([len(candidates[i]) for i in range(m)])
                else: n=self.model.sentence_len
                b=np.random.permutation(m)[:int(self.opt.max_size)//n]
                candidates=[candidates[i] for i in b]
                responding_pos= [responding_pos[i] for i in b]
                if self.opt.model=='LSTM':
                    candidate_var = text_to_var(candidates, self.src_vocab)
                    pred_probs = self.model(candidate_var)
                    log_pred_prob, best_candidate_id = pred_probs[:, 1-y].max(dim=0)
                    final_pos=responding_pos[best_candidate_id.data[0]][0]
                    final_choice=responding_pos[best_candidate_id.data[0]][1]
                    pred_prob = exp(log_pred_prob.data[0])
                else:
                    candidate_var = self.text_to_var_CNN(candidates, self.src_vocab)
                    pred_probs,_ = self.model(candidate_var)
                    log_pred_prob, best_candidate_id = pred_probs[:, 1-y].max(dim=0)
                    final_pos=responding_pos[best_candidate_id.data[0]][0]
                    final_choice=responding_pos[best_candidate_id.data[0]][1]
                    pred_prob = log_pred_prob.data[0]                        
                print('final changed pos '+str(final_pos)+' from '+sentences[final_pos]+' ------->>>>> '+list_closest_neighbors[final_pos][final_choice]+', score='+str(pred_prob))
                sentences[final_pos]=list_closest_neighbors[final_pos][final_choice]
                return sentences, final_pos, pred_prob
            else:
                return sentences, -1, 0
    def load_vocab(self, path):
        src_field = data.Field()
        label_field = data.Field(pad_token=None, unk_token=None)
        dataset = data.TabularDataset(
            path=path, format='tsv',
            fields=[('text', src_field), ('label', label_field)]
        )
        src_field.build_vocab(dataset, max_size=100000, min_freq=2, vectors="glove.6B.300d")
        label_field.build_vocab(dataset)
        return src_field.vocab, label_field.vocab

    def attack(self, count, doc, y):
        st=time.time()
        #---------------------------------word paraphrasing----------------------------------------------#

        if self.opt.model=='LSTM':
            VladsOutput = open("VladsOutputLSTM.txt", "a")
            VladsOutput.write("Original Label " + str(y) + "\n")
            VladsOutput.write("Original words: \n")
            Splitdoc = doc.split() # I basically do this because I realised that in the original emails, there are double spaces (e.g. at the start of new sentences and the adversarial examples that are outputted do not have the double spacing because Lei used words=doc.split(), so I though in order to actually use the text metric length correctly I need both original and perturbed to have single spacing.
            VladOriginalDoc = ' '.join(Splitdoc)

            VladsOutput.write(str(VladOriginalDoc))
        else:
            VladsOutput = open("VladsOutputCNN.txt", "a")
            VladsOutput.write("Original Label " + str(y) + "\n")
            VladsOutput.write("Original words: \n")
            Splitdoc = doc.split() # I basically do this because I realised that in the original emails, there are double spaces (e.g. at the start of new sentences and the adversarial examples that are outputted do not have the double spacing because Lei used words=doc.split(), so I though in order to actually use the text metric length correctly I need both original and perturbed to have single spacing.
            VladOriginalDoc = ' '.join(Splitdoc)

            VladsOutput.write(str(VladOriginalDoc))
            VladsOutput.write("\n")
            #VladsOutput.write(str(doc))

        words=doc.split()
        words_before=copy(words) # copy is used in place of == as with == if you change the new variable, it will also change the old one
        best_words=copy(words)
        # check if the value of this doc to be right

        if self.opt.model=='LSTM':
            doc_var = text_to_var([words], self.src_vocab)
        else:
            doc_var = self.text_to_var_CNN([words], self.src_vocab)
        orig_prob, orig_pred = classify(doc_var, self.model)
        pred, pred_prob = orig_pred, orig_prob

        #VladsOutput.write("This is pred " + str(pred))    #pred is either tensor(1) or tensor(0)
        #VladsOutput.write("This is y " + str(y))
        #VladsOutput.write("This is pred_prob " + str(pred_prob))
        #VladsOutput.write("This is TAU " + str(TAU))

        AttackExecuted = True


        if not (pred == y or pred_prob < TAU):
        #if (pred == y or pred_prob < TAU):
            VladsOutput.write("\nAttack failed:Initial predicted label was wrong\n")
            AttackExecuted = False

            # hence it just stops the whole attack process for that data point?
            return words, pred_prob, 0
        best_score=1-pred_prob
        # now word level paraphrasing
        list_closest_neighbors=[]
        for pos, w in enumerate(words):
            if self.opt.model=='CNN' and pos>=self.model.sentence_len: break
            try:
                closest_neighbors = self.w2v.most_similar(positive=[w.lower()], topn=N_NEIGHBOR)
            except:
                closest_neighbors=[]
            closest_paraphrases=[]
            closest_paraphrases.extend(closest_neighbors)
            # check if the words make sense
            valid_paraphrases=[]
            doc1=nlp(w)
            for repl,repl_sim in closest_paraphrases:
                doc2=nlp(repl)  #' '.join(repl_words))
                score=doc1.similarity(doc2)
                syntactic_diff = self.lm.log_prob_diff(words, pos, repl)
                logging.debug("Syntactic difference: %f", syntactic_diff)
                if score>=self.TAU_wmd_w and syntactic_diff <= self.TAU_2:
                    valid_paraphrases.append(repl)
                    
            list_closest_neighbors.append(valid_paraphrases) #closest_neighbors)
            randomcounter = 0
            if not closest_paraphrases: randomcounter += 1#neighbors: 
                #print('find no neighbor for word: '+w)
        changed_pos=set()
        iteration=0
        recompute=True
        n_change=0
        if self.opt.model=='CNN':
            lword=min(len(words), self.model.sentence_len)
        else: lword=len(words)
        while (pred == y or pred_prob < TAU) and time.time()-st<3600 \
                and n_change < self.DELTA_W * lword and len(changed_pos)+N_REPLACE<len(words):
            iteration+=1
            if recompute:  # when words are changed, the gradient wrt other words might change as well
                if self.opt.model=='LSTM':
                    doc_var = text_to_var([words], self.src_vocab)
                    embed_doc = self.model.embedding(doc_var)
                    embed_doc = Variable(embed_doc.data, requires_grad=True) # make it a leaf node and requires gradient
                    output = self.forward_lstm(embed_doc, self.model) 
                elif self.opt.model=='CNN':
                    doc_var = self.text_to_var_CNN([words], self.src_vocab)
                    embed_doc = self.model.embedding(doc_var)
                    embed_doc = embed_doc.view(embed_doc.size(0),1,-1)
                    embed_doc = Variable(embed_doc.data, requires_grad=True) # make it a leaf node and requires gradient
                    output = self.forward_cnn(embed_doc, self.model)
                if torch.cuda.is_available():
                    loss = self.criterion(output, Variable(torch.LongTensor([y])).cuda())
                else:
                    loss = self.criterion(output, Variable(torch.LongTensor([y])))
                loss.backward()
                # obtained the gradient with respect to the per word embedding, \
                # for each word, we need to compute the dot product between the embedding of each possible replacements
                # and the gradient, and replace the most negative one
                score = np.zeros(len(words)) #,1+N_NEIGHBOR*2))
                # save the score of the nearest paraphrases and the original word
                if self.opt.model=='CNN':
                    grad_data=embed_doc.grad.data[0,0,:].view(-1,300)
                for pos, w in enumerate(words):
                    if self.opt.model=='CNN' and pos>=self.model.sentence_len: break
                    if pos in changed_pos or not list_closest_neighbors[pos]:
                        continue   # don't want to change again, or if there's no choice of replacement
                    if self.opt.model=='CNN':
                        a=grad_data[pos,:]
                    else:
                        a=embed_doc.grad.data[pos,0,:].view(300)
                    score[pos]=torch.dot(a,a)
            min_score=[]
            valid_n=0
            for i in range(len(list_closest_neighbors)):
                if list_closest_neighbors[i] and not i in changed_pos:
                    min_score.append(-score[i]) 
                    valid_n+=1
                else:
                    min_score.append(10000)
            indices=np.argsort(min_score)
            if valid_n<N_REPLACE: 
                AttackExecuted = False
                VladsOutput.write("\nAttack failed: Not enough valid neighbouring words\n")
                break # does this mean if there are not enough viable words, stop the whole while loop??
            words, pred_prob, N_CHANGE=self.word_paraphrase(words, indices[:N_REPLACE], list_closest_neighbors, y)
            for i in indices[:N_CHANGE]: changed_pos.add(i)
            if pred_prob>best_score:
                best_words=copy(words)
                best_score=pred_prob
            else:
                words=copy(best_words)
                recompute=False

            if pred_prob>0.5:
                #is this the new prediction or the old one???
                #also I am pretty sure the LSTM's probabilties are not on a 0-1 scale, I think they are log numbers (e.g. some are negative)
                pred=1-y
                #VladsOutput.write("ChaNGed LaBEl " + str(pred) + "\n") #pred should be changed here (if attack was successful)
                VladsOutput.write("\nChaNGed LaBEl YESSS" + "\n") #pred should be changed here (if attack was successful)
            n_change=sum([0 if words_before[i]==words[i] else 1 for i in range(len(words))])
        dump_p_row(self.opt.output_path+'_per_word'+self.suffix+'.csv', [count, best_words, pred, pred_prob, list(changed_pos)])
        #print('after change:',' '.join(best_words),best_score)
        

        if AttackExecuted == True:

            VladAdvExample = ' '.join(best_words)
            VladsOutput.write("\nLeiAdvEx \n")
            VladsOutput.write(VladAdvExample)
            VladsOutput.write("\n")
            VladsOutput.write("NuMBer WoRds ChaNGed " + str(n_change) + "\n")
        #VladsOutput.write("Changed words \n")
        #VladsOutput.write(str(list(changed_pos)) + "\n")
            
        
        VladsOutput.write("Finished Attacking This Sample, Please Proceed" + "\n") #  I wanted to make this fairly unique so that when I read this file I in Python I can easily separate each attack

        print("This is that line 415")
        print(n_change, len(words_before), len(best_words), lword)
        return best_words, pred_prob, n_change*10.0/lword if best_words else 0

def main():
    opt = parse_args()
    X_train, y_train =read_data(opt.train_path, opt.first_label)
    X, y = read_data(opt.test_path, opt.first_label)
    attacker=Attacker(X_train,opt)
    del X_train 
    del y_train
    suc=0
    suffix='wordonly-'+str(opt.word_delta)

    Datapointcount = 0

    for count, doc in enumerate(X):
        Datapointcount += 1
        #VladsOutput = open("VladsOutputLSTM.txt", "a")
        #VladsOutput.write("This is sample number " + str(Datapointcount) + "\n") VladsOutput hasnt been created yet 
        print("this is data point " + str(Datapointcount) + " out of " + str(len(X)))
        #print("this is initial label ")
        print("this is y[count] ")
        print(y[count])
        logging.info("Processing %d/%d documents", count + 1, len(X))
        print("Processing %d/%d documents, success %d/%d", count+1, len(X), suc, count)
        changed_doc, flag, num_changed = attacker.attack(count, doc, y[count])
        #flag is actually pred_prob i believe
        try:
            v=float(flag)
            if v>0.7:
                suc+=1
                changed_y=y[count]
            else:
                changed_y=1-y[count]
        except:
            changed_y=1-y[count]
        print("this is change label ")
        print(changed_y)
        
        dump_row(opt.output_path+suffix+'.tsv', changed_doc, changed_y)
        fout = open(opt.output_path+'_count'+suffix+'.csv','a')
        fout.write(str(count)+','+str(flag)+','+str(num_changed)+'\n')
        fout.close()
    #VladsOutput.write("Overall, it was successful " + str(suc) + " times") 

if __name__ == '__main__':
    main()
