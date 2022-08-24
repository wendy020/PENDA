import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from collections import OrderedDict

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GAN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, in_channels=3,
                 out_channels=400, frames=300, alpha=0.5):
        super(GAN, self).__init__()
        
        self.alpha = alpha
        
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        
        self.l1 = nn.Sequential(
            nn.Linear(in_channels, out_channels // 4, bias=True),
            nn.BatchNorm1d(out_channels // 4),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.l2 = nn.Sequential(
            nn.Linear(out_channels // 4, out_channels // 2, bias=True),
            nn.BatchNorm1d(out_channels // 2),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.l3 = nn.Sequential(
            nn.Linear(out_channels // 2, out_channels, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.final = nn.Sequential(
            nn.Linear(out_channels, in_channels, bias=True),
            nn.Tanh()
        )
        
    def statistic(self, x):
        N, C, T, V, M = x.size() # (32, 3, 96, 16, 1) batch=32
        mins, idx = torch.min(x, 0, keepdim=True) # (1, 3, 96, 16, 1)
        maxs, idx = torch.max(x, 0, keepdim=True) # (1, 3, 96, 16, 1)
        means = torch.mean(x, 0, keepdim=True) # (1, 3, 96, 16, 1)
        
        return mins, maxs, means
    
    def MF(self, x, mins, maxs, means):
        # x: (1, 3, 96, 16, 1)
        if x <= means:
            return (x - mins) / (means - mins)
        else:
            return (maxs - x) / (maxs - means) 
        
    def frame_possibility(self, x):
        # x: (1, 3, 96, 16, 1)
        N, C, T, V, M = x.size()
        
        # drop age
        mf = x[:,:C-1,:,:,:]
        
        # multiply all [x,y,z] of all keypoints
        # (1, 3, 96, 16, 1)=>(1, 96, 16, 1)=>(1, 96, 1)=>(1, 96)
        frame_poss = torch.prod(torch.prod(mf, dim=1), dim=2).view(N, T)
        # 3*16 root
        frame_poss = torch.pow(frame_poss, 1/(C*V))
        
        return frame_poss
    
    def total_possibility(self, x):
        # x: (1, 96)
        total_poss = torch.sum(x, dim=1) / x.size(1)
        
        # output (1)
        return total_poss
    
    def validRange(self, mins, maxs, means):
        min_valid = means - (means - mins) * (1 - self.alpha)
        max_valid = means + (maxs - means) * (1 - self.alpha)
        
        return min_valid, max_valid
    
    def forward(self, x):
        mins, maxs, means = self.statistic(x)
        min_valid, max_valid = self.validRange(mins, maxs, means)
        
        N, C, T, V, M = x.size() # (1, 3, 96, 16, 1)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T) # (1, 48, 96)
        
        x = self.data_bn(x)
        
        x = x.view(N, M, V, C, T).permute(0, 1, 2, 4, 3).contiguous().view(N * M * V * T, C) # (1*1*96*16, 3)
        
        x = self.l1(x) # (1*1*96*16, 75)
        x = self.l2(x) # (1*1*96*16, 150)
        x = self.l3(x) # (1*1*96*16, 300)
        
        x = self.final(x) # (1*1*96*16, 3) [-1,1]
        
        x = x.view(N, T, V, M, C).contiguous().permute(0, 4, 1, 2, 3) # (1, 3, 96, 16, 1)
        
        # (x+1)/2 => -1+1/2=0, 1+1/2=1 => [0,1]
        # x = ((x+1)/2) * (maxs - mins) + mins # (1, 3, 96, 16, 1)
        
        x = ((x+1)/2) * (max_valid - min_valid) + min_valid # (1, 3, 96, 16, 1)
        
        return x

class CGAN(GAN):
    def __init__(self, num_class=60, num_point=25, num_person=2, in_channels=3,
                 out_channels=400, frames=300, alpha=0.5):
        super(CGAN, self).__init__(num_class, num_point, num_person, in_channels,
                                        out_channels, frames, alpha)

        self.l1 = nn.Sequential(
            nn.Linear(in_channels + 1, out_channels // 4, bias=True),
            nn.BatchNorm1d(out_channels // 4),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, label):
        mins, maxs, means = self.statistic(x)
        min_valid, max_valid = self.validRange(mins, maxs, means)
        
        N, C, T, V, M = x.size() # (1, 3, 96, 16, 1)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T) # (1, 48, 96)
        
        x = self.data_bn(x)
        
        x = x.view(N, M, V, C, T).permute(0, 1, 2, 4, 3).contiguous().view(N * M * V * T, C) # (1*1*96*16, 3)

        # concat label embedding
        # (N, T, V, M)
        label = label.view(N, T, V, M, 1).permute(0, 3, 2, 1, 4).contiguous().view(N * M * V * T, 1)
        x = torch.cat([x, label], dim=1)

        x = self.l1(x) # (1*1*96*16, 75)
        x = self.l2(x) # (1*1*96*16, 150)
        x = self.l3(x) # (1*1*96*16, 300)
        
        x = self.final(x) # (1*1*96*16, 3) [-1,1]
        
        x = x.view(N, T, V, M, C).contiguous().permute(0, 4, 1, 2, 3) # (1, 3, 96, 16, 1)
        
        # (x+1)/2 => -1+1/2=0, 1+1/2=1 => [0,1]
        # x = ((x+1)/2) * (maxs - mins) + mins # (1, 3, 96, 16, 1)
        
        x = ((x+1)/2) * (max_valid - min_valid) + min_valid # (1, 3, 96, 16, 1)
        
        return x


class Encoder(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, in_channels=3,
                 out_channels=400, frames=300):
        super(Encoder, self).__init__()

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        
        self.l1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.l2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels // 2),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.l3 = nn.Sequential(
            nn.Conv1d(out_channels // 2, out_channels // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.latent_mean = nn.Sequential(
            nn.Linear(out_channels // 4, out_channels // 8, bias=True),
            nn.Tanh()
        )

        self.latent_logvar = nn.Sequential(
            nn.Linear(out_channels // 4, out_channels // 8, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        N, C, T, V, M = x.size() # (batch, dim, frame, joint, person)
        
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)

        x = self.data_bn(x) # (N, M * V * C, T)

        x = x.view(N, M, V, C, T)
        x = x.permute(0, 3, 1, 2, 4).contiguous().view(N, C, M * V * T)
        
        x = self.l1(x) # (N, C, M * V * T)
        x = self.l2(x)
        x = self.l3(x)

        x = x.permute(0, 2, 1)
        
        mean = self.latent_mean(x)
        logvar = self.latent_logvar(x)

        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, in_channels=50,
                 out_channels=3, frames=300):
        super(Decoder, self).__init__()

        self.in_channels = 16

        self.data_bn = nn.BatchNorm1d((num_person * num_point * frames) // 8)
        
        self.l1 = nn.Sequential(
            nn.ConvTranspose1d(self.in_channels, self.in_channels * 2, kernel_size=2, stride=2),
            nn.BatchNorm1d(self.in_channels * 2),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.l2 = nn.Sequential(
            nn.ConvTranspose1d(self.in_channels * 2, self.in_channels * 4, kernel_size=2, stride=2),
            nn.BatchNorm1d(self.in_channels * 4),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.l3 = nn.Sequential(
            nn.ConvTranspose1d(self.in_channels * 4, self.in_channels * 8, kernel_size=2, stride=2),
            nn.BatchNorm1d(self.in_channels * 8),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.final = nn.Sequential(
            nn.Linear(self.in_channels * 8, 1),
            nn.Tanh()
        )

    def forward(self, x, label):
        N, C, E = x.size() # (batch, M * V * T // 8, 8)

        x = self.data_bn(x)

        x = x.view(N, C * E)

        # concat label embedding
        # (N, T, V, M)
        N, T, V, M = label.size()
        label = label.view(N, T, V, M, 1).permute(0, 3, 2, 1, 4).contiguous().view(N, M * V * T)
        
        x = torch.cat([x, label], dim=1)

        x = x.view(N, M, V, T // 8, 8, 2)
        x = x.view(N, (M * V * T) // 8, 16)
        x = x.permute(0, 2, 1)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x) # (N, -, M * V * T)

        x = x.view(N, self.in_channels * 8, M, V, T)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(N * M * V * T, self.in_channels * 8)

        x = self.final(x)

        return x

class FAG(GAN):
    def __init__(self, num_class=60, num_point=25, num_person=2, in_channels=3,
                 out_channels=400, frames=300, alpha=0.5, fuzzy=True):
        super(FAG, self).__init__(num_class, num_point, num_person, in_channels,
                                        out_channels, frames, alpha)
        self.fuzzy = fuzzy
        self.in_channels = in_channels

        self.latent_dim = out_channels // 8

        self.encoder = Encoder(num_class, num_point, num_person, in_channels,
                                out_channels, frames)
        self.decoder = Decoder(num_class, num_point, num_person, num_point * 2,
                                in_channels, frames)
        self.embedding = nn.Linear(out_channels // 8, 8)
        self.fc = nn.Linear(1, in_channels)

    def forward(self, x, label):
        N, C, T, V, M = x.size() # (batch, dim, frame, joint, person)

        mins, maxs, means = self.statistic(x)
        min_valid, max_valid = self.validRange(mins, maxs, means)

        mean, logvar = self.encode(x)
        
        latent = self.reparameterize(mean, logvar) # (N, M * V * T // 8, out_channels // 8)

        x = self.decode(latent, label) # (N * V * M * T, 1)

        x = x.view(N, 1, T, V, M) # (1, 3, 96, 16, 1)
        
        # (x+1)/2 => -1+1/2=0, 1+1/2=1 => [0,1]

        if self.fuzzy:
            x = ((x+1)/2) * (max_valid - min_valid) + min_valid # (1, 3, 96, 16, 1)
        else:
            x = self.fc(x)
            x = x.permute(0, 4, 2, 3, 1)

        return x, mean, logvar
    
    def sample(self, num_sample, label):
        x = torch.randn(num_sample, self.latent_dim)
        x.to(device)

        # concat label embedding
        # (N, T, V, M)
        N, T, V, M = label.size()
        label = label.view(N, T, V, M, 1).permute(0, 3, 2, 1, 4).contiguous().view(N, M * V * T)
        x = torch.cat([x, label], dim=1)

        x = self.decode(x)

        return x

    def loss_function(self, recons_x, x, mean, logvar, condition):
        loss = nn.MSELoss()
        MSE = loss(recons_x, x)

        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        recons_con = recons_x[:,self.in_channels-1]
        ConL = loss(recons_con, condition)

        total_loss = MSE + KLD + ConL

        return total_loss

    def encode(self, x):
        mean, logvar = self.encoder(x)

        return mean, logvar

    def decode(self, x, label):
        x = self.embedding(x)

        x = self.decoder(x, label)

        return x

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        x = eps * std + mean

        return x
