import sys

sys.path.extend(['../'])
from . import tools


num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (1, 3), (2, 4), (3, 5), (1, 6), (1, 7), (6, 7),
                    (6, 8), (7, 9), (8, 10), (9, 11), (6, 12), (7, 13),
                    (12, 13), (12, 14), (13, 15), (14, 16), (15, 17)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
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
