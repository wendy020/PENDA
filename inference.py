import glob

import torch
import torch.utils.data as data

import pandas as pd

from HAGCN.model.model import Model
from data_loader.data import *

folder = "data/3d/caiipe/skeleton_data/*/*"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    skeleton = load_data(folder)

    labels = [-1 for i in range(len(skeleton))]

    data_dim = 3
    frames = 96
    stride = 60
    data_type = "numpy"
    joint_num = 5
    batch_size = 32

    dataset = TestAIMSCDataset(file_list=skeleton, label_list=labels, dim=data_dim, max_frames=frames, stride=stride,
                           data_type=data_type, joint_num=joint_num, age_hint=True, level_hint=False, angle_hint=True, angle_diff_hint=False)

    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    model = [Model(num_class=1, num_point=joint_num, num_person=1, graph="HAGCN.graph.h36m.Graph",
                in_channels=5, out_channels=300, frames=frames) for i in range(5)]
    PATH = ["HAGCN/weights/hagcn_fold{}.pth".format(i) for i in range(5)]
    for i in range(5):
        model[i].load_state_dict(torch.load(PATH[i]))
        model[i].to(device)

    file_name = []
    predict_label = []

    with torch.no_grad():
        for i in range(5):
            model[i].eval()

        for i, data in enumerate(dataloader, 0):
            inputs, labels, names, paths, infos = data

            inputs = inputs.to(device)
            # (1, data_dim, frames, num_point, num_person)
            inputs = inputs.permute(0, 4, 2, 3, 1)

            outputs = [model[i](inputs) for i in range(5)]

            predicted = [outputs[i][:,0].clone() for i in range(5)]
            
            for i in range(5):
                predicted[i][predicted[i] >= 0.5] = 1
                predicted[i][predicted[i] < 0.5] = 0

            # name result
            for n in range(len(names)):
                file_name.append(names[n])
                p = sum([predicted[i][n].item() for i in range(5)])
                if p >= 3:
                    predict_label.append("pass")
                else:
                    predict_label.append("fail")
    
    result = {"name": file_name, "predicted": predict_label}
    print(result)

    df = pd.DataFrame(data=result)
    filepath = 'results/hagcn.xlsx'

    df.to_excel(filepath, index=False)