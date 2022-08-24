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
import torch_two_sample.statistics_diff as diff
from scipy.stats import norm
import time

from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import SubsetRandomSampler, DataLoader

from model.model import Model

import argparse
from distutils.util import strtobool
from tqdm import tqdm

import pandas as pd

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from LSTMAE.model_LSTM import *
from FLGSAN.model import GAN, CGAN, FAG
from data_loader.data import *

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# utility
def init_args():
    # add_argument use type=<type> to call <type>(arg)
    # so it don't work for bool, bool("str") always equal to True
    # use strtobool instead of bool

    parser = argparse.ArgumentParser(description='HAGCN args')
    # data loading arguments
    parser.add_argument('--train_folder', type=str, default='../data/3d/caiipe/skeleton_data', metavar='S',
                        help='train folder (default: ../data/3d/caiipe/skeleton_data)')
    parser.add_argument('--all_data', type=strtobool, default=True, metavar='B',
                        help='using pass and fail data to train (default: True)')
    parser.add_argument('--all_level', type=strtobool, default=True, metavar='B',
                        help='using all level to train (default: True)')
    parser.add_argument('--balance', type=strtobool, default=False, metavar='B',
                        help='using same amount of pass and fail data (default: False)')

    # dataset arguments
    parser.add_argument('--data_dim', type=int, default=3, metavar='N',
                        help='data dimension (default: 3)')
    parser.add_argument('--frames', type=int, default=96, metavar='N',
                        help='max frames (default: 96)')
    parser.add_argument('--stride', type=int, default=60, metavar='N',
                        help='strides of frames (default: 60)')
    parser.add_argument('--data_type', type=str, default="numpy", metavar='S',
                        help='data type (default: numpy)')
    parser.add_argument('--joint_num', type=int, default=5, metavar='N',
                        help='joint numbers (default: 5)')
    parser.add_argument('--age_hint', type=strtobool, default=True, metavar='B',
                        help='age hint (default: True)')
    parser.add_argument('--level_hint', type=strtobool, default=False, metavar='B',
                        help='level hint (default: False)')
    parser.add_argument('--angle_hint', type=strtobool, default=True, metavar='B',
                        help='mean angle hint (default: True)')
    parser.add_argument('--angle_diff_hint', type=strtobool, default=False, metavar='B',
                        help='angle difference hint (default: False)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 4)')

    # model arguments
    parser.add_argument('--in_channels', type=int, default=4, metavar='N',
                        help='in channels for generator(default: 4)')
    parser.add_argument('--in_channels_hagcn', type=int, default=5, metavar='N',
                        help='in channels for hagcn(default: 5)')                
    parser.add_argument('--out_channels', type=int, default=300, metavar='N',
                        help='out channels (default: 300)')
    parser.add_argument('--conditional', type=strtobool, default=True, metavar='B',
                        help='use conditional generator (default: True)')    
    parser.add_argument('--model', type=str, default="FL-GSAN", metavar='S',
                        help='generator model (default: FL-GSAN)')        
    parser.add_argument('--fuzzy', type=strtobool, default=True, metavar='B',
                        help='use fuzzy function (default: True)')          

    # train arguments
    parser.add_argument('--max_epoch', type=int, default=100, metavar='N',
                        help='max epoch (default: 100)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='N',
                        help='weight_decay of SGD (default: 1e-4)')
    parser.add_argument('--save_model', type=strtobool, default=False, metavar='B',
                        help='save model (default: False)')
    parser.add_argument('--model_name', type=str, default='hagcn', metavar='S',
                        help='model name (default: hagcn)')
    parser.add_argument('--data_aug', type=strtobool, default=True, metavar='B',
                        help='using data augmentation (default: True)')
    parser.add_argument('--pass_path', type=str, default='../FLGSAN/weights/flgsan_pass', metavar='S',
                        help='path of gsan pass model weight (default: ../FLGSAN/weights/flgsan_pass)')
    parser.add_argument('--fail_path', type=str, default='../FLGSAN/weights/flgsan_fail', metavar='S',
                        help='path of gsan fail model weight (default: ../FLGSAN/weights/flgsan_fail)')
    parser.add_argument('--bootstrap', type=strtobool, default=False, metavar='B',
                        help='using bootstrap sampling (default: False)')
    parser.add_argument('--fake_real_pass', type=float, default=1, metavar='F',
                        help='fake real ratio for pass data (default: 1)')
    parser.add_argument('--fake_real_fail', type=float, default=1, metavar='F',
                        help='fake real ratio for fail data (default: 1)')
    parser.add_argument('--fake_real_pass_bts', type=float, default=1, metavar='F',
                        help='fake real ratio for pass data (default: 1)')
    parser.add_argument('--fake_real_fail_bts', type=float, default=1, metavar='F',
                        help='fake real ratio for fail data (default: 1)')
    parser.add_argument('--alpha', type=float, default=0.9, metavar='N',
                        help='alpha of generator (default: 0.9)')
    parser.add_argument('--repeat_times', type=int, default=1, metavar='N',
                        help='repeat times (default: 1)')
    parser.add_argument('--gsan_new_dir', type=str, default="../data/generated_by_gsan", metavar='S',
                        help='directory where place the new data generated by FAG (default: ../data/generated_by_gsan)')    
    parser.add_argument('--init_state', type=int, default=0, metavar='N',
                        help='init random state (default: 0)')                 
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

class HAGCN_Trainer:
    def __init__(self):
        self.modalities = self.get_modalities()

        # models
        self.model = Model(num_class=1, num_point=args.joint_num, num_person=1, graph="graph.h36m.Graph",
                in_channels=args.in_channels_hagcn, out_channels=args.out_channels, frames=args.frames)
        self.model.to(device)

        if args.data_aug:
            self.generator_pass, self.generator_fail = self.load_gsan()

        # training
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
        self.criterion = nn.BCELoss()
        self.m = nn.Sigmoid()
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20,40,60,80], gamma=0.1)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60], gamma=0.5)

        # metrics
        self.acc_list = {"val": [0 for i in range(5)], "test": [0 for i in range(5)]}
        self.sen_list = {"val": [0 for i in range(5)], "test": [0 for i in range(5)]}
        self.spe_list = {"val": [0 for i in range(5)], "test": [0 for i in range(5)]}
        self.time_list = {"val": [0 for i in range(5)], "test": [0 for i in range(5)]}

        # fix random state
        self.random_state = [i+args.init_state for i in range(args.repeat_times)]

        # data
        self.mean_angles = {"age": [0 for i in range(14)], "level": [0 for i in range(3)]}

    def get_modalities(self):
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

    def load_gsan(self):
        generator_pass = []
        generator_fail = []

        if args.conditional:
            if args.model == "FL-GSAN":
                for i in range(5):
                    generator_pass.append(FAG(num_class=1, num_point=13, num_person=1, in_channels=args.in_channels,
                            out_channels=args.out_channels, frames=args.frames, alpha=args.alpha, fuzzy=args.fuzzy))
                    generator_fail.append(FAG(num_class=1, num_point=13, num_person=1, in_channels=args.in_channels,
                            out_channels=args.out_channels, frames=args.frames, alpha=args.alpha, fuzzy=args.fuzzy))
            elif args.model == "LSTM-AE":
                for i in range(5):
                    generator_pass.append(GenNet(in_dim=args.frames, Num=75))
                    generator_fail.append(GenNet(in_dim=args.frames, Num=75))
            elif args.model == "CGAN":
                for i in range(5):
                    generator_pass.append(CGAN(num_class=1, num_point=13, num_person=1, in_channels=args.in_channels,
                            out_channels=args.out_channels, frames=args.frames, alpha=args.alpha))
                    generator_fail.append(CGAN(num_class=1, num_point=13, num_person=1, in_channels=args.in_channels,
                            out_channels=args.out_channels, frames=args.frames, alpha=args.alpha))
        else:
            for i in range(5):
                generator_pass.append(GAN(num_class=1, num_point=13, num_person=1, in_channels=args.in_channels,
                                out_channels=args.out_channels, frames=args.frames, alpha=args.alpha))
                generator_fail.append(GAN(num_class=1, num_point=13, num_person=1, in_channels=args.in_channels,
                                out_channels=args.out_channels, frames=args.frames, alpha=args.alpha))
        
        for i in range(5):
            if args.model == "LSTM-AE":
                PATH_pass = args.pass_path + ".pth"
                PATH_fail = args.fail_path + ".pth"
            else:
                PATH_pass = args.pass_path + "_fold" + str(i) + ".pth"
                PATH_fail = args.fail_path + "_fold" + str(i) + ".pth"
            generator_pass[i].load_state_dict(torch.load(PATH_pass))
            generator_fail[i].load_state_dict(torch.load(PATH_fail))
            generator_pass[i].to(device)
            generator_fail[i].to(device)

        return generator_pass, generator_fail

    def data_preprocess(self, data):
        inputs, labels, names, paths, infos = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # (1, data_dim, frames, num_point, num_person)
        inputs = inputs.permute(0, 4, 2, 3, 1)

        return inputs, labels

    def parsing_path(self, path):
        name = os.path.splitext(os.path.basename(path))[0]
        ID = int(name.split('_')[0])
        age = name.split('_')[1]
        age = int(float(age.split('m')[0]))
        level = int(path.split("level")[1][:1])

        return {"id": ID, "age": age, "level": level}

    def compute_angle(self, clips):
        # (frames, num_point, data_dim)
        if args.joint_num == 5:
            nose = clips[:,2,:]
            shoulder_mid = (clips[:,3,:] + clips[:,4,:]) / 2
            hip_mid = (clips[:,0,:] + clips[:,1,:]) / 2
        else:
            nose = clips[:,6,:]
            shoulder_mid = (clips[:,7,:] + clips[:,10,:]) / 2
            hip_mid = (clips[:,0,:] + clips[:,3,:]) / 2
        
        mean_angle = 0
        for i in range(clips.shape[1]):
            mean_angle += angle(nose[i,:], shoulder_mid[i,:], hip_mid[i,:])
        mean_angle /= clips.shape[1]

        return mean_angle

    def compute_info(self, data, mode):
        mean_angles = [self.compute_angle(data[i]) for i in range(len(data))]
        infos = []
        
        for ang in mean_angles:
            # compute distance
            distance = []
            for i in range(len(self.mean_angles[mode])):
                distance.append((ang - self.mean_angles[mode][i]) * (ang - self.mean_angles[mode][i]))
            infos.append(distance.index(min(distance)))
            if mode == "age":
                infos[-1] += 1
            if mode == "level" and infos[-1] == 2:
                infos[-1] = 3
        
        return infos

    def generate_with_gsan(self, dataloader, fold):
        print("use {} as generator".format(args.model))

        aug_data = []
        aug_label = []
        for i, data in enumerate(dataloader, 0):
            inputs, labels, names, paths, infos = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            ages = infos["age"].to(device)
            # (1, data_dim, frames, num_point, num_person)
            inputs = inputs.permute(0, 4, 2, 3, 1)

            # build pass/fail list
            pass_data = {"input": [], "path": [], "age": [], "age_int": []}
            fail_data = {"input": [], "path": [], "age": [], "age_int": []}
            for j in range(len(inputs)):
                aug_data.append(paths[j])
                aug_label.append(labels[j].item())
                if labels[j] == 1:
                    pass_data["input"].append(inputs[j])
                    pass_data["path"].append(paths[j])
                    pass_data["age"].append(ages[j])
                    pass_data["age_int"].append(infos["age_int"][j])
                else:
                    fail_data["input"].append(inputs[j])
                    fail_data["path"].append(paths[j])
                    fail_data["age"].append(ages[j])
                    fail_data["age_int"].append(infos["age_int"][j])

            pass_data["input"] = torch.stack(pass_data["input"]).to(device)
            fail_data["input"] = torch.stack(fail_data["input"]).to(device)

            pass_data["age"] = torch.stack(pass_data["age"]).to(device)
            fail_data["age"] = torch.stack(fail_data["age"]).to(device)

            # generate new pass and fail skeleton
            new_pass_size = int(len(pass_data["input"])*args.fake_real_pass)
            new_fail_size = int(len(fail_data["input"])*args.fake_real_fail)

            new_pass = []
            new_fail = []
            while True:
                if len(new_pass) < new_pass_size:
                    if args.conditional:
                        if args.model == "FL-GSAN":
                            new_pass_t, mean, logvar = self.generator_pass[fold](pass_data["input"][:,:args.data_dim+1],
                                                                pass_data["age"])
                            new_pass += list(new_pass_t[:,:args.data_dim])
                        elif args.model == "LSTM-AE":
                            x = pass_data["input"][:,:args.data_dim]
                            # (batch, dim, frame, joint, person)
                            x = x.permute(0, 4, 1, 3, 2)
                            # (batch, person * dim * joint, frame)
                            N, M, C, V, T = x.size()
                            x = torch.reshape(x, (N, M * C * V, T))

                            y = self.generator_pass[fold](x)

                            y = torch.reshape(y, (N, M, C, V, T))
                            y = y.permute(0, 2, 4, 3, 1)

                            new_pass += list(y)
                        else:
                            new_pass += list(self.generator_pass[fold](pass_data["input"][:,:args.data_dim+1],
                                                                pass_data["age"])[:,:args.data_dim])
                    else:
                        new_pass += list(self.generator_pass[fold](pass_data["input"][:,:args.in_channels])[:,:args.data_dim])
                if len(new_fail) < new_fail_size:
                    if args.conditional:
                        if args.model == "FL-GSAN":
                            new_fail_t, mean, logvar = self.generator_fail[fold](fail_data["input"][:,:args.data_dim+1],
                                                                fail_data["age"])
                            new_fail += list(new_fail_t[:,:args.data_dim])
                        elif args.model == "LSTM-AE":
                            x = fail_data["input"][:,:args.data_dim]
                            # (batch, dim, frame, joint, person)
                            x = x.permute(0, 4, 1, 3, 2)
                            # (batch, person * dim * joint, frame)
                            N, M, C, V, T = x.size()
                            x = torch.reshape(x, (N, M * C * V, T))

                            y = self.generator_fail[fold](x)

                            y = torch.reshape(y, (N, M, C, V, T))
                            y = y.permute(0, 2, 4, 3, 1)

                            new_fail += list(y)
                        else:
                            new_fail += list(self.generator_fail[fold](fail_data["input"][:,:args.data_dim+1],
                                                                fail_data["age"])[:,:args.data_dim])
                    else:
                        new_fail += list(self.generator_fail[fold](fail_data["input"][:,:args.in_channels])[:,:args.data_dim])

                if len(new_pass) >= new_pass_size and len(new_fail) >= new_fail_size:
                    break
            if new_pass_size != 0:
                new_pass = torch.stack(new_pass)[:new_pass_size].permute(4, 0, 2, 3, 1)
            if new_fail_size != 0:
                new_fail = torch.stack(new_fail)[:new_fail_size].permute(4, 0, 2, 3, 1)

            if new_pass_size != 0:
                new_pass = new_pass.cpu().detach().numpy()[0]
            if new_fail_size != 0:
                new_fail = new_fail.cpu().detach().numpy()[0]
            # (1, data_dim, frames, num_point, num_person)

            # generate new file name and path
            # shuffle origin data paths
            index = [i for i in range(len(pass_data["path"]))]
            np.random.shuffle(index)
            shuffle_pass_paths = [pass_data["path"][i] for i in index]
            index = [i for i in range(len(fail_data["path"]))]
            np.random.shuffle(index)
            shuffle_fail_paths = [fail_data["path"][i] for i in index]

            # parsing id, age, level
            pass_infos = []
            fail_infos = []
            idx = 0
            for j in range(len(new_pass)):
                if j == len(shuffle_pass_paths):
                    idx = 0
                pass_infos.append(self.parsing_path(shuffle_pass_paths[idx]))
                idx += 1

            idx = 0
            for j in range(len(new_fail)):
                if j == len(shuffle_fail_paths):
                    idx = 0
                fail_infos.append(self.parsing_path(shuffle_fail_paths[idx]))
                idx += 1
            
            age_pass = pass_data["age_int"]
            age_fail = fail_data["age_int"]

            # choose level
            level_pass = self.compute_info(new_pass, "level")
            level_fail = self.compute_info(new_fail, "level")
            
            # build new paths
            new_pass_paths = ["{}/pass/level{}/{}_{}m_Pull_to_sit_generated{}.npy".
                                format(args.gsan_new_dir, level, info["id"], age, idx) for info, age, level, idx in
                                                                                            zip(pass_infos, age_pass, level_pass,
                                                                                                range(len(new_pass)))]
            new_fail_paths = ["{}/fail/level{}/{}_{}m_Pull_to_sit_generated{}.npy".
                                format(args.gsan_new_dir, level, info["id"], age, idx) for info, age, level, idx in
                                                                                            zip(fail_infos, age_fail, level_fail,
                                                                                                range(len(new_fail)))]

            # store new data
            for j in range(len(new_pass)):
                with open(new_pass_paths[j], "wb") as f: # write binary file
                    np.save(f, new_pass[j], allow_pickle=True)
            for j in range(len(new_fail)):
                with open(new_fail_paths[j], "wb") as f: # write binary file
                    np.save(f, new_fail[j], allow_pickle=True)
            
            # add new data
            for j in range(len(new_pass_paths)):
                aug_data.append(new_pass_paths[j])
                aug_label.append(1)
            for j in range(len(new_fail_paths)):
                aug_data.append(new_fail_paths[j])
                aug_label.append(0)

        # shuffle concat data
        aug_index = [i for i in range(len(aug_data))]
        np.random.shuffle(aug_index)
        aug_data = [aug_data[i] for i in aug_index]
        aug_label = [aug_label[i] for i in aug_index]

        aug_dataset = TestAIMSCDataset(file_list=aug_data, label_list=aug_label, dim=args.data_dim, max_frames=args.frames,
                                     stride=args.stride, data_type=args.data_type, joint_num=args.joint_num, age_hint=args.age_hint,
                                     level_hint=args.level_hint, angle_hint=args.angle_hint, angle_diff_hint=args.angle_diff_hint)
        print("augmented train dataset distribution(FL-GSAN)")
        print("size: {}".format(len(aug_dataset)))
        aug_dataset._distribution()
        aug_loader = DataLoader(aug_dataset, batch_size=args.batch_size, shuffle=True)

        return aug_loader

    def generate_with_bootstrap(self, dataloader):
        aug_data = []
        aug_label = []
        for i, data in enumerate(dataloader, 0):
            inputs, labels, names, paths, infos = data
            
            pass_list = []
            fail_list = []
            # add origin data
            for j in range(len(inputs)):
                aug_data.append(paths[j])
                aug_label.append(labels[j].item())
                if labels[j].item() == 1:
                    pass_list.append(j)
                elif labels[j].item() == 0:
                    fail_list.append(j)
            
            # add new data
            boot_index_pass = np.random.choice(pass_list, int(len(pass_list) * (args.fake_real_pass_bts)))
            boot_index_fail = np.random.choice(fail_list, int(len(fail_list) * (args.fake_real_fail_bts)))
            for j in boot_index_pass:
                aug_data.append(paths[j])
                aug_label.append(labels[j].item())
            for j in boot_index_fail:
                aug_data.append(paths[j])
                aug_label.append(labels[j].item())
        
        # create dataloader
        if args.data_aug:
            joint_num = 13
        else:
            joint_num = args.joint_num
        
        aug_dataset = TestAIMSCDataset(file_list=aug_data, label_list=aug_label, dim=args.data_dim, max_frames=args.frames,
                                     stride=args.stride, data_type=args.data_type, joint_num=joint_num, age_hint=args.age_hint,
                                     level_hint=args.level_hint, angle_hint=args.angle_hint, angle_diff_hint=args.angle_diff_hint,
                                     modalities=['skeleton', 'age'])
        print("augmented train dataset distribution(bootstrap)")
        print("size: {}".format(len(aug_dataset)))
        aug_dataset._distribution()
        aug_loader = DataLoader(aug_dataset, batch_size=args.batch_size, shuffle=True)

        return aug_loader

    def data_augmentation(self, dataloader, fold):
        if args.bootstrap: # bootstrap sampling
            dataloader = self.generate_with_bootstrap(dataloader)
        if args.data_aug:
            dataloader = self.generate_with_gsan(dataloader, fold)
        aug_loader = dataloader

        return aug_loader

    def compute_confusion_value(self, predicted, labels):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for p in range(len(predicted)):
            if predicted[p] == 1 and labels[p] == 1:
                TP += 1
            elif predicted[p] == 0 and labels[p] == 0:
                TN += 1
            elif predicted[p] == 1 and labels[p] == 0:
                FP += 1
            else:
                FN += 1

        return TP, TN, FP, FN

    def train_batch(self, train_loader):
        # train
        self.model.train()

        # load batch
        train_acc = 0
        train_TP = 0
        train_TN = 0
        train_FP = 0
        train_FN = 0
        total = 0
        running_loss = 0.0
        for i, train_data in enumerate(train_loader, 0):
            inputs, labels = self.data_preprocess(train_data)
            inputs = inputs[:,:args.in_channels_hagcn]
                        
            # initialize
            self.optimizer.zero_grad()

            # classify
            outputs = self.model(inputs)
            predicted = outputs[:,0].clone()
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0

            # compute loss/acc
            loss = self.criterion(self.m(outputs)[:,0], labels)
            running_loss += loss.item()

            TP, TN, FP, FN = self.compute_confusion_value(predicted, labels)
            train_TP += TP
            train_TN += TN
            train_FP += FP
            train_FN += FN

            train_acc += (predicted == labels).sum().item()
                
            total += len(predicted)


            # update weight
            loss.backward(retain_graph=True)
            self.optimizer.step()

        return train_acc, running_loss, total

    def eval_batch(self, eval_loader, fold, mode):
        # validation:
        with torch.no_grad():
            self.model.eval()
                        
            start = time.time()
            # load batch
            eval_loss = 0.0
            eval_acc = 0
            eval_TP = 0
            eval_TN = 0
            eval_FP = 0
            eval_FN = 0
            total = 0
            pass_size = 0
            fail_size = 0
            for i, eval_data in enumerate(eval_loader, 0):
                inputs, labels = self.data_preprocess(eval_data)
                inputs = inputs[:,:args.in_channels_hagcn]

                pass_size += len(labels[labels==1])
                fail_size += len(labels[labels==0])

                # inference
                outputs = self.model(inputs)
                predicted = outputs[:,0].clone()
                predicted[predicted >= 0.5] = 1
                predicted[predicted < 0.5] = 0

                # compute loss/acc
                loss = self.criterion(self.m(outputs)[:,0], labels)
                eval_loss += loss.item()

                TP, TN, FP, FN = self.compute_confusion_value(predicted, labels)
                eval_TP += TP
                eval_TN += TN
                eval_FP += FP
                eval_FN += FN

                eval_acc += (predicted == labels).sum().item()
                total += len(predicted)

            end = time.time()
            inference_time = (end - start) / total

            if mode == "val":
                if eval_acc/total >= self.acc_list[mode][fold]:
                    self.acc_list[mode][fold] = eval_acc/total
                    self.sen_list[mode][fold] = (eval_TP/(eval_TP+eval_FN))*100
                    self.spe_list[mode][fold] = (eval_TN/(eval_TN+eval_FP))*100
                    self.time_list[mode][fold] = inference_time
                    PATH = "weights/" + args.model_name + "_fold" + str(fold) + ".pth"
                    torch.save(self.model.state_dict(), PATH)
            elif mode == "test":
                self.acc_list[mode][fold] = eval_acc/total
                self.sen_list[mode][fold] = (eval_TP/(eval_TP+eval_FN))*100
                self.spe_list[mode][fold] = (eval_TN/(eval_TN+eval_FP))*100
                self.time_list[mode][fold] = inference_time
                print("confusion matrix:")
                print("\t\tTrue\tFalse")
                print("Positive\t{}\t{}".format(eval_TP, eval_FP))
                print("Negative\t{}\t{}".format(eval_TN, eval_FN))

            return eval_acc, eval_loss, total, eval_TP, eval_TN, eval_FP, eval_FN

    def loss_acc_figure(self, fold, train_acc_list, val_acc_list, train_loss_list, val_loss_list):
        epochs = [(i) for i in range(args.max_epoch)]
        plt.title("Fold {} Accuracy".format(fold))
        plt.plot(epochs, train_acc_list, label="train acc")
        plt.plot(epochs, val_acc_list, label="val acc")
        plt.legend()
        plt.savefig("log/acc_hagcn_fold{}.png".format(fold))
        plt.clf() # clear figure
            
        plt.title("Fold {} Loss".format(fold))
        plt.plot(epochs, train_loss_list, label="train loss")
        plt.plot(epochs, val_loss_list, label="val loss")
        plt.legend()
        plt.savefig("log/loss_hagcn_fold{}.png".format(fold))
        plt.clf() # clear figure

    def dataLoader2tensor(self, dataloader):
        sample_list = []
        label_list = []
        for i, data in enumerate(dataloader, 0):
            inputs, labels, names, paths, infos = data
            sample_list += list(paths)
            label_list += labels.tolist()

        return sample_list, label_list

    def train_val(self, dataloader, fold):
        # rebuild dataset
        sample_list, label_list = self.dataLoader2tensor(dataloader)
        dataset = TestAIMSCDataset(file_list=sample_list, label_list=label_list, dim=args.data_dim, max_frames=args.frames,
                                    stride=args.stride, data_type=args.data_type, joint_num=args.joint_num, age_hint=args.age_hint,
                                    level_hint=args.level_hint, angle_hint=args.angle_hint, angle_diff_hint=args.angle_diff_hint,
                                    modalities=self.modalities)

        # seperate to train/val dataset
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random.randint(0, 10))

        print("train+val dataset size: {}".format(len(dataset)))

        # create label list
        label_list = 0
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        for i, data in enumerate(dataloader, 0):
            inputs, labels, names, paths, infos = data
            label_list = labels
        
        # set train/validation dataset
        for tv_fold, (train_idx, val_idx) in enumerate(kfold.split(dataset, label_list)):
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)

            # data augmentation with the whole train dataset
            train_data_list, train_label_list = self.dataLoader2tensor(train_loader)
            # set batch size to the length of data
            joint_num = 13 if args.data_aug else args.joint_num
            train_dataset = TestAIMSCDataset(file_list=train_data_list, label_list=train_label_list, dim=args.data_dim, max_frames=args.frames,
                                    stride=args.stride, data_type=args.data_type, joint_num=joint_num, age_hint=args.age_hint,
                                    level_hint=args.level_hint, angle_hint=args.angle_hint, angle_diff_hint=args.angle_diff_hint, 
                                    modalities=self.modalities)
            for i in range(14):
                self.mean_angles["age"][i] = train_dataset.means["age"][i]
            for i in range(3):
                self.mean_angles["level"][i] = train_dataset.means["level"][i]
            
            if not args.data_aug and not args.bootstrap:
                print("train dataset distribution")
                print("size: {}".format(len(train_dataset)))
                train_dataset._distribution()
            train_loader = DataLoader(train_dataset, batch_size=len(train_data_list))
            train_loader = self.data_augmentation(train_loader, fold)
            val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler)

            train_acc_list = []
            val_acc_list = []
            train_loss_list = []
            val_loss_list = []
            best_acc = 0
            with tqdm(range(args.max_epoch)) as tepoch:
                for epoch in tepoch:
                    tepoch.set_description("Epoch {}".format(epoch))

                    # train
                    train_acc, running_loss, total = self.train_batch(train_loader)
                    train_acc_list.append((train_acc/total)*100)
                    train_loss_list.append(running_loss/total)

                    tepoch.set_postfix(loss=running_loss/total, accuracy=(train_acc/total)*100, total=total)
                    
                    self.scheduler.step()

                    # validation
                    val_acc, running_loss, total, TP, TN, FP, FN = self.eval_batch(val_loader, fold, "val")
                    val_acc_list.append((val_acc/total)*100)
                    val_loss_list.append(running_loss/total)

                # store training process
                self.loss_acc_figure(fold, train_acc_list, val_acc_list, train_loss_list, val_loss_list)

            break # use kfold to seperate train/val, not execute kfold cross-validation

    def train(self, dataset, n_time):
        # separate to fold
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state[n_time])
            
        print("whole dataset size: {}".format(len(dataset)))

        # create label list
        label_list = 0
        print("total dataset distribution")
        dataset._distribution()
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        for i, data in enumerate(dataloader, 0):
            inputs, labels, names, paths, infos = data
            label_list = labels
        
        test_acc_list = []
        test_loss_list = []
        test_TP = 0
        test_TN = 0
        test_FP = 0
        test_FN = 0
        # set train/test dataset
        for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset, label_list)):
            print("\n\nFold {}".format(fold))
            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(test_idx)
            
            train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
            test_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler)

            # train and validation
            self.train_val(train_loader, fold)

            # test
            # load best model
            PATH = "weights/" + args.model_name + "_fold" + str(fold) + ".pth"
            self.model.load_state_dict(torch.load(PATH))
            self.model.to(device)
            test_acc, running_loss, total, TP, TN, FP, FN = self.eval_batch(test_loader, fold, "test")
            test_TP += TP
            test_TN += TN
            test_FP += FP
            test_FN += FN

            print("Fold {}: Accuracy: {}, Loss: {}".format(fold, (test_acc/total)*100, running_loss/total))

        # compute summary metrics of folds
        self.acc_list["test"] = [acc*100 for acc in self.acc_list["test"]]
        M_acc = mean(self.acc_list["test"])
        print("acc :{:.2f}+-{:.2f}".format(M_acc, std(self.acc_list["test"], M_acc)))
        M_sen = mean(self.sen_list["test"])
        print("sen :{:.2f}+-{:.2f}".format(M_sen, std(self.sen_list["test"], M_sen)))
        M_spe = mean(self.spe_list["test"])
        print("spe :{:.2f}+-{:.2f}".format(M_spe, std(self.spe_list["test"], M_spe)))
        M_time = mean(self.time_list["test"])
        print("time :{:.4f}+-{:.4f}".format(M_time, std(self.time_list["test"], M_time)))

        print("confusion matrix:")
        print("\t\tTrue\tFalse")
        print("Positive\t{}\t{}".format(test_TP, test_FP))
        print("Negative\t{}\t{}".format(test_TN, test_FN))

        return M_acc, M_sen, M_spe, M_time

    def experiment(self, dataset):
        final_acc = []
        final_sen = []
        final_spe = []
        final_time = []
        for n_time in range(args.repeat_times):
            print("n_time: {}, random state: {}".format(n_time+1, self.random_state[n_time]))
            self.__init__()
            acc, sen, spe, time = self.train(dataset, n_time)
            final_acc.append(acc)
            final_sen.append(sen)
            final_spe.append(spe)
            final_time.append(time)

        print("==============================")
        print("repeat times: {}".format(args.repeat_times))
        M_acc = mean(final_acc)
        print("acc :{:.2f}+-{:.2f}".format(M_acc, std(final_acc, M_acc)))
        M_sen = mean(final_sen)
        print("sen :{:.2f}+-{:.2f}".format(M_sen, std(final_sen, M_sen)))
        M_spe = mean(final_spe)
        print("spe :{:.2f}+-{:.2f}".format(M_spe, std(final_spe, M_spe)))
        M_time = mean(final_time)
        print("time :{:.4f}+-{:.4f}".format(M_time, std(final_time, M_time)))
        print("==============================")

def randomSelect(datalist, labels):
    pass_data = [datalist[i] for i in range(len(datalist)) if labels[i] == 1]
    fail_data = [datalist[i] for i in range(len(datalist)) if labels[i] == 0]
    pass_index = np.array([i for i in range(len(pass_data))])
    select_pass = np.random.choice(np.array(pass_index), len(fail_data), replace=False).tolist()
    pass_data = [pass_data[i] for i in select_pass]
    select_data = pass_data + fail_data
    labels = [1 for i in range(len(pass_data))] + [0 for i in range(len(fail_data))]

    return select_data, labels

if __name__ == "__main__":
    args = init_args()
    
    train_folder = args.train_folder
    if args.all_data:
        train_folder = train_folder + "/*"
    if args.all_level:
        train_folder = train_folder + "/*"
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

    # args.balance = False
    if args.balance:
        sample, labels = randomSelect(sample, labels)

    dataset = TestAIMSCDataset(file_list=sample, label_list=labels, dim=args.data_dim, max_frames=args.frames, stride=args.stride,
                    data_type=args.data_type, joint_num=args.joint_num, age_hint=args.age_hint, level_hint=args.level_hint,
                    angle_hint=args.angle_hint, angle_diff_hint=args.angle_diff_hint, modalities=["skeleton"])
        
    # trainer
    trainer = HAGCN_Trainer()
    trainer.experiment(dataset)
