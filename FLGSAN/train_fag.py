import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable

import os
from collections import OrderedDict
import pickle
from glob import glob

import numpy as np
import pandas as pd
import random
import math

import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from scipy.stats import norm
import time

from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, DataLoader

import argparse
from distutils.util import strtobool
from tqdm import tqdm

from model import FAG

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from data_loader.data import *

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# utility
def init_args():
    # add_argument use type=<type> to call <type>(arg)
    # so it don't work for bool, bool("str") always equal to True
    # use strtobool instead of bool

    parser = argparse.ArgumentParser(description='gsan args')
    # data loading arguments
    parser.add_argument('--train_folder', type=str, default='../data/3d/caiipe/skeleton_data/pass', metavar='S',
                        help='train folder (default: ../data/3d/caiipe/skeleton_data/pass)')
    parser.add_argument('--all_level', type=strtobool, default=True, metavar='B',
                        help='load train data from all level (default: True)')
    parser.add_argument('--level_data', type=strtobool, default=False, metavar='B',
                        help='use level data (not AIMS) (default: False)')
    # dataset arguments
    parser.add_argument('--data_dim', type=int, default=3, metavar='N',
                        help='data dimension (default: 3)')
    parser.add_argument('--frames', type=int, default=96, metavar='N',
                        help='max frames (default: 96)')
    parser.add_argument('--stride', type=int, default=60, metavar='N',
                        help='strides of frames (default: 60)')
    parser.add_argument('--data_type', type=str, default="numpy", metavar='S',
                        help='data type (default: numpy)')
    parser.add_argument('--joint_num', type=int, default=13, metavar='N',
                        help='joint numbers (default: 13)')
    parser.add_argument('--age_hint', type=strtobool, default=True, metavar='B',
                        help='age hint (default: True)')
    parser.add_argument('--level_hint', type=strtobool, default=False, metavar='B',
                        help='level hint (default: False)')
    parser.add_argument('--angle_hint', type=strtobool, default=False, metavar='B',
                        help='angle hint (default: False)')
    parser.add_argument('--angle_diff_hint', type=strtobool, default=False, metavar='B',
                        help='angle difference hint (default: False)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')

    # model arguments
    parser.add_argument('--in_channels', type=int, default=4, metavar='N',
                        help='in channels (default: 4)')
    parser.add_argument('--out_channels', type=int, default=300, metavar='N',
                        help='out channels (default: 300)')
    parser.add_argument('--alpha', type=float, default=0.5, metavar='N',
                        help='alpha of decoder (default: 0.5)')
    parser.add_argument('--fuzzy', type=strtobool, default=True, metavar='B',
                        help='use fuzzy function (default: True)')

    # train arguments
    parser.add_argument('--max_epoch', type=int, default=200, metavar='N',
                        help='max epoch (default: 200)')
    parser.add_argument('--critic_step', type=int, default=5, metavar='N',
                        help='critic_step (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                        help='learning rate of model (default: 1e-4)')
    parser.add_argument('--beta1', type=float, default=0, metavar='N',
                        help='beta1 of Adam (default: 0)')
    parser.add_argument('--beta2', type=float, default=0.9, metavar='N',
                        help='beta2 of Adam (default: 0.9)')
    parser.add_argument('--save_model', type=strtobool, default=False, metavar='B',
                        help='save model (default: False)')
    parser.add_argument('--model_name', type=str, default='fag_pass', metavar='S',
                        help='model name (default: fag_pass)')
    parser.add_argument('--lambda_gp', type=float, default=10, metavar='N',
                        help='lambda for gradient penalty norm term (default: 10)')
    args = parser.parse_args()

    return args

def std(x, mean):
    std = 0.0
    for i in range(len(x)):
        std += (x[i] - mean)**2
    std /= len(x)
    std = math.sqrt(std)
        
    return std

def mean(x):
    mean = 0.0
    for i in range(len(x)):
        mean += x[i]
    mean /= len(x)
        
    return mean

class FAG_Trainer:
    def __init__(self, args):
        self.args = args

        # models
        self.model = FAG(num_class=1, num_point=args.joint_num, num_person=1, in_channels=args.data_dim+1,
                 out_channels=args.out_channels, frames=args.frames, alpha=args.alpha)
        self.model.to(device)

        # training
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

        # metrics
        self.best_loss = [float("inf") for i in range(5)]
        self.time_list = [0 for i in range(5)]

    def data_preprocess(self, data):
        args = self.args

        inputs, labels, names, paths, infos = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # (1, data_dim, frames, num_point, num_person)
        inputs = inputs.permute(0, 4, 2, 3, 1)

        age = infos["age"].to(device)

        return inputs, labels, age

    def train_batch(self, train_loader):
        args = self.args

        self.model.train()

        total = 0
        running_loss = 0.0
        for i, train_data in enumerate(train_loader, 0):
            inputs, labels, condition = self.data_preprocess(train_data)
            # condition (batch, frames, joint, person) normalized

            recons, mean, logvar = self.model(inputs, condition)

            self.optimizer.zero_grad()

            loss = self.model.loss_function(recons, inputs, mean, logvar, condition)
            loss.backward()

            running_loss += loss.item()

            total += len(inputs)

            self.optimizer.step()

        return running_loss, total        

    def val_batch(self, val_loader, fold):
        args = self.args

        # validation
        with torch.no_grad():
            self.model.eval()

            start = time.time()
            # load batch
            eval_loss = 0.0
            total = 0
            for i, val_data in enumerate(val_loader, 0):
                inputs, labels, condition = self.data_preprocess(val_data)

                recons, mean, logvar = self.model(inputs, condition)

                loss = self.model.loss_function(recons, inputs, mean, logvar, condition)

                eval_loss += loss.item()

                total += len(inputs)

                end = time.time()
                inference_time = (end - start) / total

            if eval_loss/total <= self.best_loss[fold]:
                self.best_loss[fold] = eval_loss/total
                self.time_list[fold] = inference_time
                if args.save_model:
                    PATH = "weights/" + args.model_name + "_fold" + str(fold) + ".pth"
                    torch.save(self.model.state_dict(), PATH)

            return eval_loss, total

    def loss_figure(self, loss_train, loss_val, fold):
        aegs = self.args

        # plot loss
        epochs = []
        for i in range(args.max_epoch):
            epochs.append(i)
        plt.title("FAG Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(epochs, loss_train, label="train loss")
        plt.plot(epochs, loss_val, label="val loss")
        plt.legend()
        plt.savefig("log/fag_loss_fold{}.png".format(fold))
        plt.clf() # clear figure

    def train(self, dataset):
        args = self.args

        # separate to fold
        kfold = KFold(n_splits=5, shuffle=True)
            
        print("dataset size: {}".format(len(dataset)))

        # set train/validation dataset
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
            val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler)

            loss_train = []
            loss_val = []
            with tqdm(range(args.max_epoch)) as tepoch:
                for epoch in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    # train
                    loss, total = self.train_batch(train_loader)
                    loss_train.append(loss/total)

                    tepoch.set_postfix(loss=loss/total, total=total)

                    # validation
                    loss, total = self.val_batch(val_loader, fold)
                    loss_val.append(loss/total)

            # store loss figure
            self.loss_figure(loss_train, loss_train, fold)

        # compute summary metrics of folds
        M_loss = mean(self.best_loss)
        print("loss :{:.2f}+-{:.2f}".format(M_loss, std(self.best_loss, M_loss)))
        M_time = mean(self.time_list)
        print("time :{:.4f}+-{:.4f}".format(M_time, std(self.time_list, M_time)))

def get_modalities():
    modalities = ["skeleton"]
    if args.age_hint:
        modalities.append("age")
    if args.level_hint:
        modalities.append("level")
    if args.angle_hint:
        modalities.append("angle")
    if args.angle_diff_hint:
        modalities.append("angle-diff")
        
    return modalities

if __name__ == "__main__":
    args = init_args()
    
    train_folder = args.train_folder
    if args.all_level:
        train_folder += "/*"
    # data
    print("load data from {}".format(train_folder))
    sample = load_data(train_folder)

    modalities = get_modalities()

    if args.level_data:
        l = int(sample[0].split("/level")[1][0:1])
        label = [l for i in range(len(sample))]
        dataset = TestLevelDataset(file_list=sample, label_list=label, dim=args.data_dim, max_frames=args.frames, stride=args.stride,
                           data_type=args.data_type, joint_num=args.joint_num, age_hint=args.age_hint, level_hint=args.level_hint,
                           angle_hint=args.angle_hint, angle_diff_hint=args.angle_diff_hint, modalities=modalities)
    else:
        label = [1 for i in range(len(sample))]
        
        dataset = TestAIMSCDataset(file_list=sample, label_list=label, dim=args.data_dim, max_frames=args.frames, stride=args.stride,
                           data_type=args.data_type, joint_num=args.joint_num, age_hint=args.age_hint, level_hint=args.level_hint,
                           angle_hint=args.angle_hint, angle_diff_hint=args.angle_diff_hint, modalities=modalities)
        
    # trainer
    trainer = FAG_Trainer(args)
    trainer.train(dataset)
