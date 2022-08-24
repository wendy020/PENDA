from torchvision.models import resnet50
from thop import profile
import torch

from HAGCN.model.model import Model
from FLGSAN.model import FAG
from LSTMAE.model_LSTM import *

model = Model(num_class=1, num_point=5, num_person=1, graph="HAGCN.graph.h36m.Graph",
                in_channels=5, out_channels=300, frames=96)
input = torch.randn(1, 5, 96, 5, 1)
macs, params = profile(model, inputs=(input, ))
print("HA-GCN:")
print("macs:{:.2f}G, params:{:.2f}M".format(macs/10**9, params/10**6))

model = FAG(num_class=1, num_point=13, num_person=1, in_channels=4,
                        out_channels=300, frames=96, alpha=0.5, fuzzy=True)
# (batch, dim, frame, joint, person)
input = torch.randn(1, 4, 96, 13, 1)
label = torch.randn(1, 96, 13, 1)
macs, params = profile(model, inputs=(input, label))
print("FL-GSAN:")
print("macs:{:.2f}G, params:{:.2f}M".format(macs/10**9, params/10**6))

model = GenNet(96, 75)
# (batch, dim, frame, joint, person)
input = torch.randn(1, 39, 96)
macs, params = profile(model, inputs=(input, ))
print("LSTM-AE:")
print("macs:{:.2f}G, params:{:.2f}M".format(macs/10**9, params/10**6))