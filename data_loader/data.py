import torch
import torch.utils.data as data

import os
from collections import OrderedDict
import pickle
from glob import glob

import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

from scipy.stats import norm

import sys

import pandas as pd

from matplotlib import animation

# dim = 2

# ax = plt.axes(projection='3d') if dim == 3 else plt.axes()
# scatter, = ax.plot3D(0, 0, 0, 'ro') if dim == 3 else ax.plot(0, 0, 'ro')
# bone, = ax.plot3D(0, 0, 0) if dim == 3 else ax.plot(0, 0)

# labeling the x-axis and y-axis
# axis = plt.axes(xlim=(0, 1000),  ylim=(0, 1000))
  
# lists storing x and y values
# x, y = [], []
  
# line, = axis.plot(0, 0)
fig, ax = plt.subplots()

scatter, = plt.plot([], [], 'ro')
bone, = plt.plot([], [])
# text = ax.text(0, 0, "", color='red')

def init():
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return scatter,

class TestAIMSCDataset(data.Dataset):
    def __init__(self, file_list, label_list, dim=2, max_frames=48, stride=30, data_type="numpy", joint_num=16, age_hint=False, level_hint=False,
                    angle_hint=False, angle_diff_hint=False, modalities=['skeleton']):
        super(TestAIMSCDataset, self).__init__()
        self.dim = dim
        self.max_frames = max_frames
        self.data_type = data_type
        self.joint_num = joint_num
        self.age_hint = age_hint
        self.level_hint = level_hint
        self.angle_hint = angle_hint
        self.angle_diff_hint = angle_diff_hint
        self.modalities = modalities
        
        self.clips = []
        self.labels = label_list
        self.subject_names = []
        self.file_paths = []
        self.ages = []
        self.age_int = []
        self.levels = []
        self.angles = []
        self.angle_diffs = []
        self.re_order_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]

        self.dist = {"0": [0, 0], "1": [0, 0], "3": [0, 0]} # [pass, fail]
        self.means = {"age": [0 for i in range(14)], "level": [0 for i in range(3)]}
        self.age_count = [0 for i in range(14)]
        self.min_coords = np.zeros((self.max_frames, self.joint_num, self.dim))
        self.max_coords = np.zeros((self.max_frames, self.joint_num, self.dim))
        
        # keypoint
        # 0    3    6    7         10
        # RHip LHip nose LShoulder RShoulder
        if joint_num == 5:
            self.joint_index = [0, 3, 6, 7, 10]
        elif joint_num == 3:
            self.joint_index = [6, 7, 10, 0, 3]
            
        self.prepare_data(file_list)
                   
    def __getitem__(self, index):
        clips = torch.tensor(np.array(self.clips[index]), dtype=torch.float)
        label = torch.tensor(self.labels[index], dtype=torch.float)
        subject_name = self.subject_names[index]
        file_path = self.file_paths[index]

        infos = {}
        
        for info in self.modalities:
            if info == "age":
                infos["age"] = torch.tensor(self.ages[index], dtype=torch.float)
                infos["age_int"] = self.age_int[index]
            elif info == "level":
                infos["level"] = torch.tensor(self.levels[index], dtype=torch.float)
            elif info == "angle":
                infos["angle"] = torch.tensor(self.angles[index], dtype=torch.float)
            elif info == "angle-diff":
                infos["angle-diff"] = torch.tensor(self.angle_diffs[index], dtype=torch.float)
        
        return clips, label, subject_name, file_path, infos
        # return clips, subject_name

    def __len__(self):
        # return len(self.labels)
        return len(self.subject_names)

    def _distribution(self):
        print("=========================================")
        print("level 0|pass: {}\t|fail: {}".format(self.dist["0"][0], self.dist["0"][1]))
        print("level 1|pass: {}\t|fail: {}".format(self.dist["1"][0], self.dist["1"][1]))
        print("level 3|pass: {}\t|fail: {}".format(self.dist["3"][0], self.dist["3"][1]))
        age_dist = "month"
        for i in range(14):
            age_dist += "|{}: {}\t".format(i+1, self.age_count[i])
        print(age_dist)
        # print("mean angle:")
        # age_angle = "age"
        # for i in range(14):
        #     age_angle += "|{}:{}\t".format(i+1, self.means["age"][i])
        # print(age_angle)
        # level_angle = "level"
        # for i in range(3):
        #     level_angle += "|{}:{}\t".format(i, self.means["level"][i])
        # print(level_angle)
        # print("joint")
        # print("frame 1")
        # for i in range(self.joint_num):
        #     min_str = "{}: ".format(i+1)
        #     for j in range(self.dim):
        #         min_str += "\t{}|{}~{}".format(j+1, self.min_coords[0][i][j], self.max_coords[0][i][j])
        #     print(min_str)
        # print("\nframe 96")
        # for i in range(self.joint_num):
        #     min_str = "{}: ".format(i+1)
        #     for j in range(self.dim):
        #         min_str += "\t{}|{}~{}".format(j+1, self.min_coords[-1][i][j], self.max_coords[-1][i][j])
        #     print(min_str)
        print("=========================================")
    
    def getinfo(self, pkl_filename):
        name = os.path.splitext(os.path.basename(pkl_filename))[0]
        ID = int(name.split('_')[0])
        age = name.split('_')[1]
        age = int(float(age.split('m')[0]))
        level = -1
        if self.level_hint:
            level = int(pkl_filename.split("level")[1][:1])
            level = 2 if level == 3 else level

        return name, ID, age, level

    def normalize_angle(self, ang, c):
        ang = 360 - ang
        if ang < 90: # decline
            ang = 90-(c*0.5)
            c+=1
        elif ang > 240: # decline
            ang = 240-(c*0.5)
            c+=1
        
        return ang, c

    def compute_angle_and_offset(self, skeleton_dict):
        skeleton_np = np.zeros((len(skeleton_dict), self.joint_num, self.dim))
        head_angle = []
        norm_angle = []
        norm_count = 0
        for i in range(len(skeleton_dict)):
            if self.dim == 2 and self.data_type != "numpy":
                sk = skeleton_dict[i][self.re_order_indices]
            elif self.joint_num == 5:
                sk = skeleton_dict[i][self.joint_index]
            elif self.joint_num == 3:
                sk = skeleton_dict[i][self.joint_index]
                sk = [sk[0], (sk[1] + sk[2]) / 2, (sk[3] + sk[4]) / 2]
            else:
                sk = skeleton_dict[i]
            if self.joint_num == 5:
                ang = angle(sk[2], (sk[0]+sk[1])/2, (sk[3]+sk[4])/2)
                norm_ang, norm_count = self.normalize_angle(ang, norm_count)
                if norm_ang < 0:
                    norm_ang = norm_angle[-1] + 0.5
                head_angle.append(ang)
                norm_angle.append(norm_ang)
                G = (skeleton_dict[i][0] + skeleton_dict[i][1]) / 2
            elif self.joint_num == 3:
                ang = angle(sk[0], sk[1], sk[2])
                norm_ang, norm_count = self.normalize_angle(ang, norm_count)
                if norm_ang < 0:
                    norm_ang = norm_angle[-1] + 0.5
                head_angle.append(ang)
                norm_angle.append(norm_ang)
                G = (skeleton_dict[i][0] + skeleton_dict[i][1]) / 2
            else:
                ang = angle(sk[2], sk[7], sk[10])
                norm_ang, norm_count = self.normalize_angle(ang, norm_count)
                if norm_ang < 0:
                    norm_ang = norm_angle[-1] + 0.5
                head_angle.append(ang)
                norm_angle.append(norm_ang)
                G = skeleton_dict[i][7]
                
            skeleton_np[i] = sk - G

        return skeleton_np, head_angle, norm_angle

    def fix_frame(self, skeleton_dict, skeleton_np, head_angle, norm_angle):
        angles = []
        n_angles = []

        # fix the number of frames
        frame_data = np.zeros((self.max_frames, self.joint_num, self.dim))

        if self.max_frames < len(skeleton_dict): # drop last k frames with 1 step
            step = len(skeleton_dict) // self.max_frames
            compensation = len(skeleton_dict) - (self.max_frames * step)
            j, k = 0, 0
            for i in range(len(skeleton_dict)):
                if(i == j):
                    frame_data[k] = skeleton_np[i]
                    angles.append(head_angle[i])
                    n_angles.append(norm_angle[i])
                    if k < (self.max_frames - compensation):
                        j = j + step
                    else:
                        j = j + (step + 1)
                    k += 1
        else: # padding with first k frames
            step = self.max_frames // len(skeleton_dict)
            compensation = self.max_frames - (len(skeleton_dict) * step)
            j, repeat_times = 0, 0
            for i in range(self.max_frames):
                frame_data[i] = skeleton_np[j]
                angles.append(head_angle[j])
                n_angles.append(norm_angle[j])
                repeat_times += 1
                if j < compensation:
                    if (repeat_times % (step + 1)) == 0:
                        repeat_times = 0
                        j += 1
                else:
                    if (repeat_times % step) == 0:
                        repeat_times = 0
                        j += 1
    
        return frame_data, angles, n_angles

    def compute_angle_diff(self, angles):
        angle_diff_np = np.zeros((self.max_frames, self.joint_num, 1))

        # initialize first angle difference
        angle_diff_np[0] = np.zeros((self.joint_num, 1))

        # compute angle difference between 2 frames
        for i in range(len(angles)-1):
            angle_diff_np[i+1] = np.ones((self.joint_num, 1)) * (angles[i+1] - angles[i])

        # standarilize
        max_diff = np.max(angle_diff_np, axis=(0, 1))
        min_diff = np.min(angle_diff_np, axis=(0, 1))
        angle_diff_np = (angle_diff_np - min_diff) / (max_diff - min_diff)

        return angle_diff_np
        
    def animate(self, index, data, v_id):
        self.display_skeleton(data, index, v_id)

        return scatter, bone #, text

    def display_skeleton(self, data, index, v_id):
        frame = []
        plt.title("skeleton {} frame {}({})".format(v_id, index, self.subject_names[v_id]))
        
        if self.dim == 2:
            plt.plot(data[index,:,0], data[index,:,1], 'bo')
            # scatter.set_data(data[index,:,0], data[index,:,1])
        elif self.dim == 3:
            plt.plot(data[index,:,0], data[index,:,1], data[index,:,2], 'bo')
            # scatter.set_data(data[index,:,0], data[index,:,1], data[index,:,2])
        if self.joint_num == 5:
            # 0    1    2    3         4
            # rhip lhip nose lshoulder rshoulder
            line = [[2, 3], [2, 4], [3, 4], [3, 1], [4, 0], [0, 1]]
            point = ["rhip", "lhip", "nose", "lshoulder", "rshoulder"]
        elif self.joint_num == 13:
            # 0    1     2     3    4     5     6    7         8      9      10        11     12
            # rhip rknee rfoot lhip lknee lfoot nose lshoulder lelbow lwrist rshoulder relbow rwrist
            line = [[6, 7], [6, 10], [7, 10], [7, 3], [10, 0], [0, 3],
                    [7, 8], [8, 9], [10, 11], [11, 12],
                    [0, 1], [1, 2], [3, 4], [4, 5]]
            point = ["rhip", "rknee", "rfoot", "lhip", "lknee", "lfoot", "nose", "lshoulder",
                    "lelbow", "lwrist", "rshoulder", "relbow", "rwrist"]
        for l in line:
            x = [data[index,l[0],0], data[index,l[1],0]]
            y = [data[index,l[0],1], data[index,l[1],1]]
            # print("x: {}".format(x))
            if self.dim == 2:
                plt.plot(x, y)
                # bone.set_data(x, y)
            elif self.dim == 3:
                z = [data[index,l[0],2], data[index,l[1],2]]
                plt.plot3D(x, y, z)
                # bone.set_data(x, y, z)
        for i in range(len(point)):
            x = data[index,i,0]
            y = data[index,i,1]
            if self.dim == 2:
                # text.set_text(x, y, point[i])
                plt.annotate(point[i], (x, y))
            elif self.dim == 3:
                z = data[index,i,2]
                # text.set_text(x, y, z, point[i])
                plt.annotate(point[i], (x, y, z))
        
        plt.savefig("../log/skeleton_{}_frame_{}_test.png".format(v_id, index))
        plt.clf() # clear figure

    def prepare_data(self, file_list):
        index = 0
        sk_np_list = []
       
        for pkl_filename in file_list:
            name, ID, age, level = self.getinfo(pkl_filename)
            self.age_int.append(age)
            self.age_count[age-1]+=1
            if level == 0:
                if self.labels[index] == 1:
                    self.dist["0"][0]+=1
                else:
                    self.dist["0"][1]+=1
            elif level == 1:
                if self.labels[index] == 1:
                    self.dist["1"][0]+=1
                else:
                    self.dist["1"][1]+=1
            else:
                if self.labels[index] == 1:
                    self.dist["3"][0]+=1
                else:
                    self.dist["3"][1]+=1

            # read skeletons
            if self.data_type == "numpy":
                skeleton_dict = np.load(pkl_filename)
            else:
                rfile = open(pkl_filename, "rb")
                skeleton_dict = pickle.load(rfile)

            skeleton_dict = skeleton_dict[:,:13] # drop 2 additional joints

            if skeleton_dict.shape[2] == 96:
                skeleton_dict = skeleton_dict.reshape(96, 32, 3)[:, [1,2,3,6,7,8,14,17,18,19,25,26,27]]
            # print(pkl_filename)
            # print(skeleton_dict.shape)
            skeleton_np, head_angle, norm_angle = self.compute_angle_and_offset(skeleton_dict)

            skeleton_np, angles, n_angles = self.fix_frame(skeleton_dict, skeleton_np, head_angle, norm_angle)
            # show_angle = "angles: "
            # for i in range(len(angles)):
            #     show_angle += "{:.1f} ".format(angles[i])
            # print(show_angle)

            sk_np_list.append(skeleton_np)
            max_coord = np.max(skeleton_np, axis=(0, 1))
            min_coord = np.min(skeleton_np, axis=(0, 1))
            skeleton_np = (skeleton_np - min_coord) / (max_coord - min_coord)

            mean_angle = sum(n_angles) / self.max_frames
            self.means["age"][age-1] += mean_angle
            self.means["level"][level] += mean_angle

            if self.age_hint:
                age_np = np.ones((self.max_frames, self.joint_num, 1)) * age / 14.
                skeleton_np = np.concatenate((skeleton_np, age_np), axis=2)
                self.ages.append(age_np)
            if self.level_hint:
                level_np = np.ones((self.max_frames, self.joint_num, 1)) * level / 3.
                skeleton_np = np.concatenate((skeleton_np, level_np), axis=2)
                self.levels.append(level_np)
            if self.angle_hint:
                mean_angle_np = np.ones((self.max_frames, self.joint_num, 1)) * mean_angle
                skeleton_np = np.concatenate((skeleton_np, mean_angle_np), axis=2)
                self.angles.append(mean_angle_np)
            if self.angle_diff_hint:
                angle_diff_np = self.compute_angle_diff(n_angles)
                skeleton_np = np.concatenate((skeleton_np, angle_diff_np), axis=2)
                self.angle_diffs.append(angle_diff_np)
            
            skeletons = []
            skeletons.append(skeleton_np)
                       
            self.clips.append(skeletons)
            self.subject_names.append(name)
            self.file_paths.append(pkl_filename)

            index+=1

        for i in range(14):
            if self.age_count[i] != 0:
                self.means["age"][i] /= self.age_count[i]
        levels = ["0", "1", "3"]
        for i in range(3):
            level_size = self.dist[levels[i]][0] + self.dist[levels[i]][1]
            if level_size != 0:
                self.means["level"][i] /= level_size
        sk_nps = np.array(sk_np_list)
        self.min_coords = np.min(sk_nps, axis=0) # (frames, joint, dim)
        self.max_coords = np.max(sk_nps, axis=0)

        # self.display_skeleton(sk_nps[2], 10, 2)
        # self.display_skeleton(sk_nps[2], 20, 2)
        # self.display_skeleton(sk_nps[2], 30, 2)
        # self.display_skeleton(sk_nps[2], 40, 2)
        # self.display_skeleton(sk_nps[2], 50, 2)

        # fig = plt.figure()
        # anim = animation.FuncAnimation(fig, self.animate, frames=96, init_func=init, 
        #                        fargs=(sk_nps[10], 0), blit=True)
        # fig.suptitle('Straight Line plot', fontsize=14)
        
        # saving to m4 using ffmpeg writer
        # writervideo = animation.FFMpegWriter(fps=60)
        # anim.save('../log/infant.mp4', writer=writervideo)
        # plt.close()

class TestLevelDataset(data.Dataset):
    def __init__(self, file_list, label_list, dim=2, max_frames=48, stride=30, data_type="numpy", joint_num=16, age_hint=False, level_hint=False,
                    angle_hint=False, angle_diff_hint=False, modalities=['skeleton']):
        super(TestLevelDataset, self).__init__()
        self.dim = dim
        self.max_frames = max_frames
        self.data_type = data_type
        self.joint_num = joint_num
        self.age_hint = age_hint
        self.level_hint = level_hint
        self.angle_hint = angle_hint
        self.angle_diff_hint = angle_diff_hint
        self.modalities = modalities
        
        self.clips = []
        self.labels = label_list
        self.subject_names = []
        self.file_paths = []
        self.ages = []
        self.levels = []
        self.angles = []
        self.angle_diffs = []
        self.re_order_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]

        self.dist = {"0": 0, "1": 0, "3": 0}
        self.means = {"age": [0 for i in range(14)], "level": [0 for i in range(3)]}
        self.age_count = [0 for i in range(14)]
        
        # keypoint
        # 0    3    6    7         10
        # RHip LHip nose LShoulder RShoulder
        if joint_num == 5:
            self.joint_index = [0, 3, 6, 7, 10]
            
        self.prepare_data(file_list)
                   
    def __getitem__(self, index):
        clips = torch.tensor(np.array(self.clips[index]), dtype=torch.float)
        label = torch.tensor(self.labels[index], dtype=torch.float)
        subject_name = self.subject_names[index]
        file_path = self.file_paths[index]

        infos = {}
        
        for info in self.modalities:
            if info == "age":
                infos["age"] = torch.tensor(self.ages[index], dtype=torch.float)
            elif info == "level":
                infos["level"] = torch.tensor(self.levels[index], dtype=torch.float)
            elif info == "angle":
                infos["angle"] = torch.tensor(self.angles[index], dtype=torch.float)
            elif info == "angle-diff":
                infos["angle-diff"] = torch.tensor(self.angle_diffs[index], dtype=torch.float)
        
        return clips, label, subject_name, file_path, infos
        # return clips, subject_name

    def __len__(self):
        # return len(self.labels)
        return len(self.subject_names)

    def _distribution(self):
        print("=========================================")
        print("level 0: {}".format(self.dist["0"]))
        print("level 1: {}".format(self.dist["1"]))
        print("level 3: {}".format(self.dist["3"]))
        age_dist = "month"
        for i in range(14):
            age_dist += "|{}: {}\t".format(i+1, self.age_count[i])
        print(age_dist)
        print("=========================================")
    
    def getinfo(self, pkl_filename):
        name = os.path.splitext(os.path.basename(pkl_filename))[0]
        ID = name.split('_')[0]
        if len(ID) > 3:
            ID = 0
        else:
            ID = int(ID)
        if name.find("_") == -1:
            age = 1
        else:
            age = name.split('_')[1]
            age = float(age.split('m')[0])
            if age > 14:
                age = 1
            else:
                age = int(age)
        # ../Kuei_level_data/origin/level{}
        # ../Kuei_level_data/fold{}/level{}
        level = int(pkl_filename.split("/level")[1][:1])
        level = 2 if level == 3 else level

        return name, ID, age, level

    def normalize_angle(self, ang, c):
        ang = 360 - ang
        if ang < 90: # decline
            ang = 90-(c*0.5)
            c+=1
        elif ang > 240: # decline
            ang = 240-(c*0.5)
            c+=1
        
        return ang, c

    def compute_angle_and_offset(self, skeleton_dict):
        skeleton_np = np.zeros((len(skeleton_dict), self.joint_num, self.dim))
        head_angle = []
        norm_angle = []
        norm_count = 0
        for i in range(len(skeleton_dict)):
            if self.joint_num == 5:
                sk = skeleton_dict[i][self.joint_index]
            else:
                sk = skeleton_dict[i]
            if self.joint_num == 5:
                ang = angle(sk[2], (sk[0]+sk[1])/2, (sk[3]+sk[4])/2)
                norm_ang, norm_count = self.normalize_angle(ang, norm_count)
                if norm_ang < 0:
                    norm_ang = norm_angle[-1] + 0.5
                head_angle.append(ang)
                norm_angle.append(norm_ang)
                G = (skeleton_dict[i][0] + skeleton_dict[i][1]) / 2
            else:
                ang = angle(sk[2], sk[7], sk[10])
                norm_ang, norm_count = self.normalize_angle(ang, norm_count)
                if norm_ang < 0:
                    norm_ang = norm_angle[-1] + 0.5
                head_angle.append(ang)
                norm_angle.append(norm_ang)
                G = skeleton_dict[i][7]
                
            skeleton_np[i] = sk - G

        return skeleton_np, head_angle, norm_angle

    def fix_frame(self, skeleton_dict, skeleton_np, head_angle, norm_angle):
        angles = []
        n_angles = []

        # fix the number of frames
        frame_data = np.zeros((self.max_frames, self.joint_num, self.dim))

        if self.max_frames < len(skeleton_dict): # drop last k frames with 1 step
            step = len(skeleton_dict) // self.max_frames
            compensation = len(skeleton_dict) - (self.max_frames * step)
            j, k = 0, 0
            for i in range(len(skeleton_dict)):
                if(i == j):
                    frame_data[k] = skeleton_np[i]
                    angles.append(head_angle[i])
                    n_angles.append(norm_angle[i])
                    if k < (self.max_frames - compensation):
                        j = j + step
                    else:
                        j = j + (step + 1)
                    k += 1
        else: # padding with first k frames
            step = self.max_frames // len(skeleton_dict)
            compensation = self.max_frames - (len(skeleton_dict) * step)
            j, repeat_times = 0, 0
            for i in range(self.max_frames):
                frame_data[i] = skeleton_np[j]
                angles.append(head_angle[j])
                n_angles.append(norm_angle[j])
                repeat_times += 1
                if j < compensation:
                    if (repeat_times % (step + 1)) == 0:
                        repeat_times = 0
                        j += 1
                else:
                    if (repeat_times % step) == 0:
                        repeat_times = 0
                        j += 1
    
        return frame_data, angles, n_angles

    def compute_angle_diff(self, angles):
        angle_diff_np = np.zeros((self.max_frames, self.joint_num, 1))

        # initialize first angle difference
        angle_diff_np[0] = np.zeros((self.joint_num, 1))

        # compute angle difference between 2 frames
        for i in range(len(angles)-1):
            angle_diff_np[i+1] = np.ones((self.joint_num, 1)) * (angles[i+1] - angles[i])

        # standarilize
        max_diff = np.max(angle_diff_np, axis=(0, 1))
        min_diff = np.min(angle_diff_np, axis=(0, 1))
        angle_diff_np = (angle_diff_np - min_diff) / (max_diff - min_diff)

        return angle_diff_np

    def prepare_data(self, file_list):
        index = 0
        for pkl_filename in file_list:
            name, ID, age, level = self.getinfo(pkl_filename)
            self.age_count[age-1]+=1
            if level == 0:
                self.dist["0"]+=1
            elif level == 1:
                self.dist["1"]+=1
            else:
                self.dist["3"]+=1

            # read skeletons
            skeleton_dict = np.load(pkl_filename)

            skeleton_np, head_angle, norm_angle = self.compute_angle_and_offset(skeleton_dict)

            skeleton_np, angles, n_angles = self.fix_frame(skeleton_dict, skeleton_np, head_angle, norm_angle)
            # show_angle = "angles: "
            # for i in range(len(angles)):
            #     show_angle += "{:.1f} ".format(angles[i])
            # print(show_angle)

            max_coord = np.max(skeleton_np, axis=(0, 1))
            min_coord = np.min(skeleton_np, axis=(0, 1))
            skeleton_np = (skeleton_np - min_coord) / (max_coord - min_coord)

            # mean_angle = sum(n_angles) / self.max_frames
            mean_angle = sum(angles) / self.max_frames
            self.means["age"][age-1] += mean_angle
            self.means["level"][level] += mean_angle

            if self.age_hint:
                age_np = np.ones((self.max_frames, self.joint_num, 1)) * age / 14.
                skeleton_np = np.concatenate((skeleton_np, age_np), axis=2)
                self.ages.append(age_np)
            if self.level_hint:
                level_np = np.ones((self.max_frames, self.joint_num, 1)) * level / 3.
                skeleton_np = np.concatenate((skeleton_np, level_np), axis=2)
                self.levels.append(level_np)
            if self.angle_hint:
                mean_angle_np = np.ones((self.max_frames, self.joint_num, 1)) * mean_angle
                skeleton_np = np.concatenate((skeleton_np, mean_angle_np), axis=2)
                self.angles.append(mean_angle_np)
            if self.angle_diff_hint:
                angle_diff_np = self.compute_angle_diff(n_angles)
                skeleton_np = np.concatenate((skeleton_np, angle_diff_np), axis=2)
                self.angle_diffs.append(angle_diff_np)
            
            skeletons = []
            skeletons.append(skeleton_np)
                       
            self.clips.append(skeletons)
            self.subject_names.append(name)
            self.file_paths.append(pkl_filename)

            index+=1

        for i in range(14):
            if self.age_count[i] != 0:
                self.means["age"][i] /= self.age_count[i]
        levels = ["0", "1", "3"]
        for i in range(3):
            level_size = self.dist[levels[i]]
            if level_size != 0:
                self.means["level"][i] /= level_size

def angle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    if ang < 0:
        ang += 360

    return ang

def load_data(input_dir):
    subjects = glob(input_dir + "/*")
    print("data size: {}".format(len(subjects)))
    return subjects

if __name__ == "__main__":
    if len(sys.argv) > 3:
        print("invalid arguments!")
        print("usage: python data.py [--test] [--level]")
    if sys.argv[1][2:] == "test":
        print("test")
        input_dir = "../data/3d/caiipe/skeleton_data/*/*"
     
        sample = load_data(input_dir)

        labels = []
        for i in range(len(sample)):
            label = sample[i].split('/')[5]
            if label == "pass":
                labels.append(1)
            elif label == "fail":
                labels.append(0)
        
        data_dim = 3
        frames = 96
        stride = 60
        data_type = "numpy"
        joint_num = 13
        batch_size = 128

        dataset = TestAIMSCDataset(file_list=sample, label_list=labels, dim=data_dim, max_frames=frames, stride=stride,
                           data_type=data_type, joint_num=joint_num, age_hint=True, level_hint=False, angle_hint=True, angle_diff_hint=True,
                           modalities=['skeleton', 'age'])
        dataset._distribution()
        dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

        data_list = {i: [] for i in range(15)}
        label_list = {i: [0, 0] for i in range(15)}
        name_list = []
        id_list = []

        for i, data in enumerate(dataloader, 0):
            inputs, labels, names, paths, infos = data
            
            for i in range(5):
                print("[{}] file: {}, label: {}".format(i, names[i], labels[i]))
            
            break
