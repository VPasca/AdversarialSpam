import argparse

import torch
from torchtext import data
from sklearn.metrics import accuracy_score

import os
#import argparse

#import torch
import torch.nn as nn
import torch.optim as optim
#from torchtext import data
#from sklearn.metrics import accuracy_score
from lstm import LSTMClassifier
#from evaluate import evaluate
import torch.nn.functional as F


device = -1
if torch.cuda.is_available():
    device = None

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

#padding=2

            if use_cuda:
                component = component.cuda()

            conv_blocks.append(component)

            '''conv_blocks.append(
                nn.Sequential(
                    conv1d,
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size = maxpool_kernel_size)
                ).cuda()
            )'''
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
        out = F.dropout(out, p=0.3, training=self.training)
        return F.softmax(self.fc(out), dim=1), feature_extracted


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('train_path', action='store', help='Path to train data')
    parser.add_argument('test_path', action='store', help='Path to test data')
    parser.add_argument('model_path', action='store', help='Path to pre-trained classifier model')
    parser.add_argument('--batch-size', action='store', default=16, type=int,
                        help='Mini batch size.')

    #, dest="model_path"
    return parser.parse_args()

def evaluate(model, dataset, batch_size):
    iterator = data.BucketIterator(
            dataset=dataset,
            batch_size=batch_size,
            device=device,
            train=False,
            sort=False # frome https://github.com/pytorch/text/issues/474
            #sort_key=lambda x:len(x.comment_text)
            #sort_within_batch=False,
            
    )
    model.eval()

    predlist = []
    labellist = []
    LabelFile = open("TrecLabelsCnn.txt","a") 
    PredFile = open("CnnPred.txt","a") 
    for batch in iterator:
        pred = model(batch.text)
        predlist += pred.max(dim=1)[1].data.tolist()
        labellist += batch.label.view(-1).data.tolist()
        
    #PredFile.writelines(L) for L = predlist
    #LabelFile.writelines(L) for L = labellist   
    
    PredFile.writelines(predlist)
    LabelFile.writelines(labellist)

    PredFile.close() 
    LabelFile.close() 
    return accuracy_score(labellist, predlist)

def main():

    #opt = parse_args()

    train_path="../data/train.tsv"
    test_path="../data/test.tsv"
    model_path="../model/model_cnn"
    batch_size=16

    src_field = data.Field()
    label_field = data.Field(pad_token=None, unk_token=None)
    train = data.TabularDataset(
        path=train_path, format='tsv',
        fields=[('text', src_field), ('label', label_field)]
    )
    test = data.TabularDataset(
        path=test_path, format='tsv',
        fields=[('text', src_field), ('label', label_field)]
    )
    src_field.build_vocab(train, max_size=100000, min_freq=1, vectors="glove.6B.300d")
    label_field.build_vocab(train)

    #classifier = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
    classifier = torch.load(model_path)

    #if torch.cuda.is_available():
    #    classifier.cuda()

    test_accu = evaluate(classifier, test, batch_size)

    print("Test accuracy: %f", test_accu)

if __name__ == '__main__':
    main()
