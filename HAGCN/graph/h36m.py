import sys

sys.path.extend(['../'])
from . import tools


num_node = 16
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1,0), (2,1), (3,2), (4,0), (5,4), (6,5), (0,7), (8,7),
                (10,8), (11,8), (12,11), (13,12), (14,8), (15, 14), (16,15)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, joint_num, labeling_mode='spatial', kp_16_3D=False):
        self.num_node = joint_num
        if self.num_node == 13:
            self.self_link = [(i, i) for i in range(self.num_node)]
            inward_ori_index = [(2,1), (1,0), (0,10), (5,4), (4,3), (3,7),
                (12,11), (11,10), (10,6), (9,8), (8, 7), (7,6)]
            self.inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
            self.outward = [(j, i) for (i, j) in self.inward]
            self.neighbor = self.inward + self.outward
        elif kp_16_3D:
            self.self_link = [(i, i) for i in range(self.num_node)]
            inward_ori_index = [(2,1), (1,0), (0,6), (5,4), (4,3), (3,6), (6,7), (15,14),
                            (14,13), (13,7), (12, 11), (11,10), (10,7), (9,8), (8,7)]
            self.inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
            self.outward = [(j, i) for (i, j) in self.inward]
            self.neighbor = self.inward + self.outward
        elif self.num_node == 5:
            self.self_link = [(i, i) for i in range(self.num_node)]
            inward_ori_index = [(0,4), (1,3), (2,4), (2,3)]
            self.inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
            self.outward = [(j, i) for (i, j) in self.inward]
            self.neighbor = self.inward + self.outward
        elif self.num_node == 3:
            self.self_link = [(i, i) for i in range(self.num_node)]
            # nose mshoulders mhips
            inward_ori_index = [(0,1), (2,1)]
            self.inward = [(i - 1, j - 1) for (i,j) in inward_ori_index]
            self.outward = [(j, i) for (i, j) in self.inward]
            self.neighbor = self.inward + self.outward
        else:
            self.self_link = self_link
            self.inward = inward
            self.outward = outward
            self.neighbor = neighbor
            
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
