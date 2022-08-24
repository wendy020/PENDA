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

from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, DataLoader

from model import GAN, CGAN, FAG

import argparse
from distutils.util import strtobool
from tqdm import tqdm

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from data_loader.data import *
from HAGCN.model.model import Model

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# utility
def init_args():
    # add_argument use type=<type> to call <type>(arg)
    # so it don't work for bool, bool("str") always equal to True
    # use strtobool instead of bool

    parser = argparse.ArgumentParser(description='FLGSAN args')
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
    parser.add_argument('--conditional', type=strtobool, default=True, metavar='B',
                        help='use conditional generator (default: True)')  
    parser.add_argument('--model', type=str, default="FL-GSAN", metavar='S',
                        help='generative model (default: FL-GSAN)')  
    parser.add_argument('--fuzzy', type=strtobool, default=True, metavar='B',
                        help='use fuzzy function (default: True)')

    # train arguments
    parser.add_argument('--max_epoch', type=int, default=200, metavar='N',
                        help='max epoch (default: 200)')
    parser.add_argument('--critic_step', type=int, default=5, metavar='N',
                        help='critic_step (default: 5)')
    parser.add_argument('--lr_G', type=float, default=1e-4, metavar='N',
                        help='learning rate of generator (default: 1e-4)')
    parser.add_argument('--lr_D', type=float, default=1e-4, metavar='N',
                        help='learning rate of discriminator (default: 1e-4)')
    parser.add_argument('--beta1', type=float, default=0, metavar='N',
                        help='beta1 of Adam (default: 0)')
    parser.add_argument('--beta2', type=float, default=0.9, metavar='N',
                        help='beta2 of Adam (default: 0.9)')
    parser.add_argument('--save_model', type=strtobool, default=False, metavar='B',
                        help='save model (default: False)')
    parser.add_argument('--model_name', type=str, default='flgsan_pass', metavar='S',
                        help='model name (default: flgsan_pass)')
    parser.add_argument('--lambda_gp', type=float, default=10, metavar='N',
                        help='lambda for gradient penalty norm term (default: 10)')
    parser.add_argument('--path', type=str, default="weights/fag_pass", metavar='N',
                        help='path of FAG (default: fag_pass.pth)')
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

class FLGSAN_Trainer:
    def __init__(self, args):
        self.args = args

        # models
        self.generator, self.discriminator = self.init_model()

        # training
        self.optimizer_G = [torch.optim.Adam(self.generator[i].parameters(), lr=args.lr_G, betas=(args.beta1, args.beta2))
                                    for i in range(5)]
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr_D, betas=(args.beta1, args.beta2))

        # metrics
        self.best_loss_G = [float("inf") for i in range(5)]
        self.best_loss_D = [float("inf") for i in range(5)]
        self.time_list = [0 for i in range(5)]

    def init_model(self):
        args = self.args

        print("use {} model".format(args.model))
        
        in_channels = args.data_dim + sum([args.age_hint, args.level_hint, args.angle_hint, args.angle_diff_hint])
        
        # Initialize generator and discriminator
        generator = []
        if args.conditional:
            if args.model == "FL-GSAN":
                for i in range(5):
                    generator.append(FAG(num_class=1, num_point=args.joint_num, num_person=1, in_channels=in_channels,
                                out_channels=args.out_channels, frames=args.frames, fuzzy=args.fuzzy))
                    PATH = args.path + "_fold" + str(i) + ".pth"
                    generator[i].load_state_dict(torch.load(PATH))

                    for param in generator[i].encoder.parameters():
                        param.requires_grad = False

            elif args.model == "CGAN":
                for i in range(5):
                    generator.append(CGAN(num_class=1, num_point=args.joint_num, num_person=1, in_channels=in_channels,
                                out_channels=args.out_channels, frames=args.frames))
        else:
            for i in range(5):
                generator.append(GAN(num_class=1, num_point=args.joint_num, num_person=1, in_channels=in_channels,
                            out_channels=args.out_channels, frames=args.frames))
        
        discriminator = Model(num_class=1, num_point=args.joint_num, num_person=1, graph="HAGCN.graph.h36m.Graph",
                    in_channels=in_channels, out_channels=args.out_channels, frames=args.frames)
        # replace batch normalization with layer normalization
        # input shape = (batch_size, num_person * in_channels * num_point, frames)
        discriminator.data_bn = nn.LayerNorm([1 * in_channels * args.joint_num, args.frames])

        for i in range(5):
            generator[i] = generator[i].to(device)
        discriminator = discriminator.to(device)

        return generator, discriminator

    def data_preprocess(self, data):
        args = self.args

        inputs, labels, names, paths, infos = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # (1, data_dim, frames, num_point, num_person)
        inputs = inputs.permute(0, 4, 2, 3, 1)

        if args.conditional:
            age = infos["age"].to(device)

            return inputs, labels, age
        else:
            return inputs, labels

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        args = self.args

        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1, 1))).to(device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates) # already cat condition to inputs
        fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_batch(self, train_loader, fold):
        args = self.args

        self.generator[fold].train()
        self.discriminator.train()

        total = 0
        loss_G = 0.0
        loss_D = 0.0
        for i, train_data in enumerate(train_loader, 0):
            if args.conditional:
                inputs, labels, condition = self.data_preprocess(train_data)
                # condition (batch, frames, joint, person) normalized
            else:
                inputs, labels = self.data_preprocess(train_data)
                    
            # Configure input
            real_skeleton = inputs

            # ---------------------
            #  Train Discriminator
            # ---------------------

            self.optimizer_D.zero_grad()

            # Generate a batch of images
            if args.conditional:
                if args.model == "FL-GSAN":
                    fake_skeleton, mean, logvar = self.generator[fold](real_skeleton, condition)
                else:
                    fake_skeleton = self.generator[fold](real_skeleton, condition)
            else:
                fake_skeleton = self.generator[fold](real_skeleton)

            # Real images
            real_validity = self.discriminator(real_skeleton) # already cat condition to inputs
            # Fake images
            fake_validity = self.discriminator(fake_skeleton) # already cat condition to inputs
            # Gradient penalty
            # already cat condition to inputs
            gradient_penalty = self.compute_gradient_penalty(self.discriminator, real_skeleton, fake_skeleton)
            # wasserstein distance=-torch.mean(real_validity) + torch.mean(fake_validity)
            wasserstein_d = -torch.mean(real_validity) + torch.mean(fake_validity)
            # Adversarial loss
            if args.conditional:
                closs = nn.MSELoss()
                Closs = closs(fake_skeleton[:,args.in_channels-1], condition)
                d_loss = wasserstein_d + args.lambda_gp * gradient_penalty + Closs
            else:
                d_loss = wasserstein_d + args.lambda_gp * gradient_penalty

            d_loss.backward()
            self.optimizer_D.step()

            self.optimizer_G[fold].zero_grad()

            # Train the generator every critic steps
            if i % args.critic_step == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                if args.conditional:
                    if args.model == "FL-GSAN":
                        fake_skeleton, mean, logvar = self.generator[fold](real_skeleton, condition)
                    else:
                        fake_skeleton = self.generator[fold](real_skeleton, condition)
                else:
                    fake_skeleton = self.generator[fold](real_skeleton)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = self.discriminator(fake_skeleton) # already cat condition to inputs
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                self.optimizer_G[fold].step()
                        
                loss_G += g_loss.item()
                loss_D += d_loss.item()

                total += len(fake_validity)

        return loss_G, loss_D, total        

    def val_batch(self, val_loader, fold):
        args = self.args

        # validation
        with torch.no_grad():
            self.generator[fold].eval()
            self.discriminator.eval()

            start = time.time()
            # load batch
            loss_G = 0.0
            loss_D = 0.0
            total = 0
            for i, val_data in enumerate(val_loader, 0):
                if args.conditional:
                    inputs, labels, condition = self.data_preprocess(val_data)
                else:
                    inputs, labels = self.data_preprocess(val_data)

                # Configure input
                real_skeleton = inputs

                # Generate a batch of images
                if args.conditional:
                    if args.model == "FL-GSAN":
                        fake_skeleton, mean, logvar = self.generator[fold](real_skeleton, condition)
                    else:
                        fake_skeleton = self.generator[fold](real_skeleton, condition)
                else:
                    fake_skeleton = self.generator[fold](real_skeleton)

                # Real images
                real_validity = self.discriminator(real_skeleton) # already cat condition to inputs
                # Fake images
                fake_validity = self.discriminator(fake_skeleton) # already cat condition to inputs

                g_loss = -torch.mean(fake_validity)

                wasserstein_d = -torch.mean(real_validity) + torch.mean(fake_validity)

                # Adversarial loss
                d_loss = wasserstein_d

                loss_G += g_loss.item()
                loss_D += d_loss.item()

                total += len(fake_validity)

                end = time.time()
                inference_time = (end - start) / total

            if loss_G/total <= self.best_loss_G[fold]:
                self.best_loss_G[fold] = loss_G/total
                self.best_loss_D[fold] = loss_D/total
                self.time_list[fold] = inference_time
                if args.save_model:
                    PATH = "weights/" + args.model_name + "_fold" + str(fold) + ".pth"
                    torch.save(self.generator[fold].state_dict(), PATH)

            return loss_G, loss_D, total

    def loss_figure(self, loss_G_train, loss_G_val, loss_D_train, loss_D_val, fold):
        aegs = self.args

        # plot loss
        epochs = []
        for i in range(args.max_epoch):
            epochs.append(i)
        plt.title("Generator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(epochs, loss_G_train, label="train loss_G")
        plt.plot(epochs, loss_G_val, label="val loss_G")
        plt.legend()
        plt.savefig("log/loss_G_fold{}.png".format(fold))
        plt.clf() # clear figure
        
        plt.title("Discriminator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(epochs, loss_D_train, label="train loss_D")
        plt.plot(epochs, loss_D_val, label="val loss_D")
        plt.legend()
        plt.savefig("log/loss_D_fold{}.png".format(fold))


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

            loss_G_train = []
            loss_D_train = []
            loss_G_val = []
            loss_D_val = []
            with tqdm(range(args.max_epoch)) as tepoch:
                for epoch in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    # train
                    loss_G, loss_D, total = self.train_batch(train_loader, fold)
                    loss_G_train.append(loss_G/total)
                    loss_D_train.append(loss_D/total)

                    tepoch.set_postfix(loss_G=loss_G/total, loss_D=loss_D/total, total=total)

                    # validation
                    loss_G, loss_D, total = self.val_batch(val_loader, fold)
                    loss_G_val.append(loss_G/total)
                    loss_D_val.append(loss_D/total)
                
                print("save model {}".format(args.model_name))

            # store loss figure
            self.loss_figure(loss_G_train, loss_G_val, loss_D_train, loss_D_val, fold)

        # compute summary metrics of folds
        M_loss_G = mean(self.best_loss_G)
        print("loss_G :{:.2f}+-{:.2f}".format(M_loss_G, std(self.best_loss_G, M_loss_G)))
        M_loss_D = mean(self.best_loss_D)
        print("loss_D :{:.2f}+-{:.2f}".format(M_loss_D, std(self.best_loss_D, M_loss_D)))
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
    trainer = FLGSAN_Trainer(args)
    trainer.train(dataset)
