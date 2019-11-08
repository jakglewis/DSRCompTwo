###to evaluate with the test data, after running create data python file
import PIL
import numpy as np
import pandas as pd
import torch
from easyimages import EasyImageList
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from skorch import *
from skorch.callbacks import (
    Callback,
    CyclicLR,
    Freezer,
    LRScheduler,
    PrintLog,
    scoring,
)
from skorch.utils import *
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import SGD
import os
import pickle

import pandas as pd
import os.path
import datetime
from pandas import datetime


# read-in pickled test data 

path_df = "4.Data/X_bow_test.pkl"
with open(path_df, 'rb') as data:
    X_bow_test = pickle.load(data) 


path_df = "4.Data/X_TFIDF_test.pkl"
with open(path_df, 'rb') as data:
    X_TFIDF_test = pickle.load(data)
# read in pickled test labels

path_df = "4.Data/Y_test.pkl"
with open(path_df, 'rb') as data:
    Y_test = pickle.load(data) 
    
    
    
# read in pickled nn-models




###evaluate nn model

Y_test = torch.tensor(Y_test.values)
X_TFIDF_test = torch.tensor(X_TFIDF_test.values)
Y_test = torch.reshape(Y_test,(len(Y_test),1))

X=X_TFIDF_test.float()
Y_test=Y_test.float()

##


NUM_LABELS=1
VOCAB_SIZE=10221


class BoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self,num_labels, vocab_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(BoWClassifier, self).__init__()
        
        self.layer1=nn.Sequential(
            nn.Linear(10221,1000),
            nn.ReLU() 
        )
        
        self.layer2=nn.Sequential(
            nn.Linear(1000,200),
            nn.ReLU() 
        )
        self.layer3=nn.Sequential(
            nn.Linear(200,64),
            nn.ReLU() 
        )
               
        self.layer4=nn.Sequential(
            nn.Linear(64,1),
            nn.Sigmoid() 
        )
#
    def forward(self, x):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        
        return out

model = BoWClassifier(1,10221)
model.load_state_dict(torch.load('model_2.pt'))


preds=[]
actual=[]
logprob = []
for instance, label in zip(X, Y_test):
    bow_vec = torch.tensor(instance)
    logprobs = model(bow_vec)
    #print(logprobs)
    pred=1 if logprobs>0.5 else 0
    #print('prediction: {}'.format([pred]))
    #print('actual: {}'.format(label))
    preds.append(pred)
    actual.append(label)
    logprob.append(logprobs)

    
from sklearn.metrics import roc_auc_score

TFIDF_metric = roc_auc_score(Y_test, preds)
print("TFIDF_NN_MODEL accuracy {}: ".format(TFIDF_metric))

