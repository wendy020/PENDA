#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 11:41:03 2017

@author: Damily
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset,DataLoader
from data import loadview_data, loadsubject_data
from model_LSTM import ActNet,GenNet
import time
from torch.optim.lr_scheduler import StepLR
import os

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from data_loader.data import *

####################################### 
import argparse
parser = argparse.ArgumentParser(description="Compress Num")
parser.add_argument('--Num',type=int,default=75)
args = parser.parse_args()
Num = args.Num
# dir_file = 'model/subject/compress_{}'.format(Num)
dir_file = 'compress_{}'.format(Num)
if not(os.path.exists(dir_file)):
   os.mkdir(dir_file)
######################################
print("*********")
batch_size = 256

train_folder = "../data/3d/caiipe/skeleton_data/*/*"

# data
print("load data from {}".format(train_folder))
sample = load_data(train_folder)

labels = []
for i in range(len(sample)):
    label = sample[i].split('/')[5]
    if label == "pass":
        labels.append(1)
    elif label == "fail":
        labels.append(0)

dsets = TestAIMSCDataset(file_list=sample, label_list=labels, dim=3, max_frames=96, stride=60,
                    data_type="numpy", joint_num=13, age_hint=False, level_hint=False,
                    angle_hint=False, angle_diff_hint=False, modalities=["skeleton"])
######################################################################
allloss = []
def train_model(model_ft, model_gen, criterion, MSEdis, optimizer, scheduler, num_epochs=60):
    since = time.time()
    model_ft.train(True) 
    model_gen.train(True) 
    dset_sizes = len(dsets)
    for epoch in range(num_epochs):
        print('Data Size',dset_sizes)           
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        dset_loaders =  DataLoader(dataset=dsets, num_workers=4,batch_size= batch_size, shuffle=True)
        # Each epoch has a training and validation phase
        # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        count = 0

        # Iterate over data.
        for data in dset_loaders:
            # get the inputs
            inputs, labels, names, paths, infos = data
            labels,inputs = Variable(labels.cuda()),Variable(inputs.cuda())

            inputs = inputs.permute(0, 1, 4, 3, 2)  

            N, M, C, V, T = inputs.size()
            inputs = torch.reshape(inputs, (N, M * C * V, T))
       
            # zero the parameter gradients
            optimizer.zero_grad()           
            #train 
            Deinputs  = model_gen(inputs)
            De_class  = model_ft(Deinputs)
            out_class = model_ft(inputs)

            De_class = De_class[:,0].clone()
            De_class[De_class >= 0.5] = 1
            De_class[De_class < 0.5] = 0

            out_class = out_class[:,0].clone()
            out_class[out_class >= 0.5] = 1
            out_class[out_class < 0.5] = 0
            
            loss_De_class = criterion(De_class,labels)
            
            loss_out_class = criterion(out_class,labels)
            
            loss_mse = MSEdis(Deinputs, inputs)
            
            loss  = loss_De_class + loss_out_class + loss_mse

            
            loss.backward()
            optimizer.step() 
            
            
            preds = De_class.data
            # backward + optimize only if in training phase
            count +=1
            if count%10==0 or inputs.size()[0]<batch_size:
                print('Epoch:{}:loss_De_class:{:.3f} loss_out_class:{:.3f} loss_mse:{:.3f}'
                    .format(epoch,loss_De_class.item(), loss_out_class.item(), loss_mse.item())) 
                allloss.append(loss.item())
				
            # statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / dset_sizes
        epoch_acc = running_corrects / (dset_sizes*1.0)

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
        if (epoch+1)%50==0 :
            model_out_path = dir_file + "/ClassLSTM_epoch_{}.pth".format(epoch)
            torch.save(model_ft.state_dict(), model_out_path)
            model_out_path = dir_file + "/GenLSTM_epoch_{}.pth".format(epoch)
            torch.save(model_gen.state_dict(), model_out_path)

        scheduler.step()
                
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return allloss
######################################################################
in_dim = 96

model_ft = ActNet(in_dim, class_num=1).cuda()
model_gen = GenNet(in_dim, Num).cuda()

criterion = nn.BCELoss().cuda()
MSEdis = nn.MSELoss().cuda()

optimizer = optim.Adam(list(model_ft.parameters())+list(model_gen.parameters()),lr=0.001)
scheduler = StepLR(optimizer,step_size=400,gamma=0.1)
######################################################################
# Train
allloss = train_model(model_ft, model_gen, criterion, MSEdis, optimizer, scheduler, num_epochs=500)
######################################
