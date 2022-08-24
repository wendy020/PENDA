import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F
# from torch.legacy.nn import CMul // Pytorch version <= 0.4.1


class ActNet(nn.Module):
    def __init__(self, in_dim, class_num):
        super(ActNet, self).__init__()

        self.bn = nn.BatchNorm1d(39, affine=False)
        self.lstm = nn.LSTM(in_dim, 100, 3, batch_first=True,dropout=0.5)
        self.linear = nn.Sequential(nn.Linear(100, class_num),nn.ELU())
   
    def forward(self, inputs):
        inputs = self.bn(inputs)
        features,_ = self.lstm(inputs)
        out = self.linear(features[:,-1,:])
        return out  
    
class GenNet(nn.Module):
    def __init__(self, in_dim, Num):
        super(GenNet, self).__init__() 
        
        self.bn = nn.BatchNorm1d(39, affine=False)
        self.Enlstm = nn.LSTM(in_dim, Num, 2, batch_first=True,dropout=0.5)
        self.Delstm = nn.LSTM(Num, in_dim, 2, batch_first=True,dropout=0.5)
   
    def forward(self, inputs):
        inputs = self.bn(inputs)
        encoder,_ = self.Enlstm(inputs)
        decoder,_ = self.Delstm(encoder) 
        return decoder
 
