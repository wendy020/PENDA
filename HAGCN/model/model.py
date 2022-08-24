import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from collections import OrderedDict

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1):
        super(unit_tcn, self).__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        
        self.tcn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                      padding=(pad, 0), stride=(stride, 1), dilation=(dilation, 1)),
            nn.BatchNorm2d(out_channels),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
                
    def forward(self, x):
        out = self.tcn(x)
        return out


class atcn(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, embed_factor=5):
        super(atcn, self).__init__()
        
        embed_channels = out_channels // embed_factor
        
        self.conv1 = unit_tcn(in_channels, embed_channels, 3, 1, 1)
        self.conv2 = unit_tcn(embed_channels, embed_channels, 3, 1, 2)
        self.conv3 = unit_tcn(embed_channels, embed_channels, 3, 1, 3)
        self.conv4 = unit_tcn(embed_channels, embed_channels, 3, 1, 4)

        self.conv5 = unit_tcn(in_channels, embed_channels, 1, 1, 1)
        
        
        self.conv_final = nn.Sequential(
            nn.Conv2d(embed_channels * 5, out_channels, kernel_size=1, groups=5),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            unit_tcn(out_channels, out_channels, 9, stride, 1),
        )
        
        self.down = unit_tcn(in_channels, out_channels, 9, stride, 1)
        
        self.relu = nn.ReLU(inplace=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))  # shape: N, out_channels, T, V
        x2 = self.relu(self.conv2(x1))  # shape: N, out_channels, T, V
        x3 = self.relu(self.conv3(x2))  # shape: N, out_channels, T, V
        x4 = self.relu(self.conv4(x3))  # shape: N, out_channels, T, V

        N, C, T, V = x1.size()
        x5 = x.mean(2, True)  # shape: N, in_channels, 1, V
        x5 = F.interpolate(x5, size=(T, V), mode="bilinear", align_corners=False)  # shape: N, in_channels, T, V
        x5 = self.relu(self.conv5(x5))  # shape: N, out_channels, T, V
        
        out = torch.cat([x1, x2, x3, x4, x5], 1)  # shape: N, out_channels*5, T, V
        out = self.conv_final(out)
        
        out = out + self.down(x)
        return self.relu(out)


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=5):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        
        self.PA1 = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        self.PA2 = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA1, 1.)
        nn.init.constant_(self.PA2, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = A.shape[0]

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size() # (3, 1, 96, 16)
        if x.get_device() >= 0:
            A = self.A.to(x.get_device())
        else:
            A = self.A
        A = A * self.PA1 + self.PA2

        y = None
        for i in range(self.num_subset):
            A1 = A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V)) # (1, 3, 96, 16)
            y = z + y if y is not None else z

        y = self.bn(y)
        y = y + self.down(x)
        return self.relu(y)

    
class self_attention_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, coff_embedding=5, frames=300, num_subset=3):
        super(self_attention_gcn, self).__init__()
        self.inter_c = out_channels // coff_embedding
        self.num_subset = num_subset
        
        self.conv_t1 = nn.ModuleList()
        self.conv_t2 = nn.ModuleList()
        self.conv_q = nn.ModuleList()
        self.conv_k = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_t1.append(nn.Conv2d(frames, 1, 1))
            self.conv_t2.append(nn.Conv2d(frames, 1, 1))
            self.conv_q.append(nn.Conv2d(in_channels, self.inter_c, 1))
            self.conv_k.append(nn.Conv2d(in_channels, self.inter_c, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        N, C, T, V = x.size()
        
        y = None
        A = x.permute(0, 2, 1, 3) # N T C V
        for i in range(self.num_subset):
            Q = self.conv_t1[i](A).permute(0, 2, 1, 3) # N C 1 V
            Q = self.conv_q[i](Q).permute(0, 3, 1, 2).contiguous().view(N, V, -1) # N V C, query
            K = self.conv_t2[i](A).permute(0, 2, 1, 3) # N C 1 V
            K = self.conv_k[i](K).view(N, -1, V) # N C V, key

            A1 = self.soft(torch.matmul(Q, K) / Q.size(-1))  # N V V, attention (adjacency matrix)
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        
        y = self.bn(y)
        y = y + self.down(x)
        return self.relu(y)
    

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, frames=300):
        super(TCN_GCN_unit, self).__init__()
        self.gcn = unit_gcn(in_channels, out_channels, A)
        self.sagcn = self_attention_gcn(in_channels, out_channels, coff_embedding=4, frames=frames)
        self.tcn = atcn(out_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=False)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, stride=stride)
        
    def forward(self, x):
        out = self.gcn(x) + self.sagcn(x)
        out = self.tcn(out)
        
        out = out + self.residual(x)
        out = self.relu(out)
        return out


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None,
                 graph_args=dict(), in_channels=3, out_channels=400, frames=300, kp_16_3D=False):
        super(Model, self).__init__()

        if graph is None:
            Graph = import_class('graph.ntu_rgb_d.Graph')
            self.graph = Graph(num_point, 'spatial', kp_16_3D)
        else:
            Graph = import_class(graph)
            self.graph = Graph(num_point, 'spatial', kp_16_3D)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        
        self.l1 = TCN_GCN_unit(in_channels, out_channels // 4, A, stride=2, residual=False, frames=frames)
        self.l2 = TCN_GCN_unit(out_channels // 4, out_channels // 2, A, stride=2, frames=frames//2)
        self.l3 = TCN_GCN_unit(out_channels // 2, out_channels, A, stride=2, frames=frames//4)

        self.fc = nn.Linear(out_channels, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size() # (1, 3, 96, 16, 1)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T) # (1, 48, 96)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V) # (1, 3, 96, 16)
        
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)