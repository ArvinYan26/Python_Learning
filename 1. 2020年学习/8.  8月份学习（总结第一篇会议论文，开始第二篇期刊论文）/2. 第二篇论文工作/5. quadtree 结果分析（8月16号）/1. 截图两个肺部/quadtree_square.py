# -*- encoding:utf8 -*-

import numpy as np
import matplotlib.image as mpimg
import cv2
import math


class Node(object):

    def __init__(self, value, isLeaf, left_1, left_2, right_1, right_2, pos, grid_size):
        """
        Args:
            val:  (mean,variance)
            isLeaf:
            pos:(top left point)
            size : n: (n*n)
        """
        self.value = value
        self.isLeaf = isLeaf
        self.left_1 = left_1
        self.left_2 = left_2
        self.right_1 = right_1
        self.right_2 = right_2

        self.pos = pos
        self.grid_size = grid_size


class QuadTreeSolution(object):
    def __init__(self, image_grid, min_grid_size,
                 variance_threshold={1024: 100, 512: 100, 256: 400, 128: 250, 64: 200, 32: 25, 16: 50, 8: 20000, 4: 20,
                                     2: 20,
                                     1: 20}, ):
        """

        Args:
            image_grid:   img np array
            variance_threshold:
        """

        self.image_grid = image_grid
        self.variance_threshold = variance_threshold
        self.min_grid_size = min_grid_size
        self.hisogram = {}
        self.image_size = len(image_grid)
        #self.variance_map = {2 ^ i: v for i in range(10)}
        if isinstance(variance_threshold, int):
            self.variance_map = {2 ** i: variance_threshold for i in range(11)}
        else:

            self.variance_map = variance_threshold
        self.root_node = self.node_construct(image_grid, pos=(0, 0), grid_size=self.image_size)

        # self.dic_order_by_key()

    def dic_order_by_key(self):
        """  hisogram: dict -> ordered list (高到低排序)
            [(256, 2), (128, 15), (64, 69), (32, 223), (16, 390), (8, 616), (4, 1344)]
        Returns:

        """
        res = []
        keys_list = sorted(self.hisogram, reverse=True)  # H -> L
        for key in keys_list:
            res.append((key, self.hisogram[key]))
        self.hisogram = res

    def node_construct(self, grid, pos, grid_size):
        """
        Args:
            grid: numpy Array, (m,m)
            variance_threshold: the variance in a grid, the threshold for segmentation grid.
            pos : [(top left point),(right bottom point)]

        Returns:
            node: root Node
        """
        "Parameters Initialization"
        grid_mean = np.mean(grid)
        grid_variance = np.var(grid)
        # print(grid_variance)

        value = (grid_mean, grid_variance)
        isleaf = True
        left_1, left_2, right_1, right_2 = None, None, None, None
        #if grid_size <= self.min_grid_size or grid_variance <= self.variance_threshold:
        if grid_size <= self.min_grid_size or grid_variance <= self.variance_map.get(grid_size, 20):
            if self.hisogram.get(grid_size) == None:
                self.hisogram[grid_size] = 1
            else:
                self.hisogram[grid_size] += 1
            return Node(value, isleaf, left_1, left_2, right_1, right_2, pos=pos, grid_size=grid_size)
        assert grid_size % 2 == 0
        center_point = grid_size // 2
        left_1_grid = grid[:center_point, :center_point]
        left_2_grid = grid[:center_point, center_point:]
        right_1_grid = grid[center_point:, :center_point]
        right_2_grid = grid[center_point:, center_point:]
        left_1 = self.node_construct(left_1_grid, pos=pos, grid_size=center_point)
        left_2 = self.node_construct(left_2_grid, pos=(pos[0] + center_point, pos[1]), grid_size=center_point)
        right_1 = self.node_construct(right_1_grid, pos=(pos[0], pos[1] + center_point), grid_size=center_point)
        right_2 = self.node_construct(right_2_grid, pos=(pos[0] + center_point, pos[1] + center_point),
                                      grid_size=center_point)
        value = False
        isleaf = False
        node = Node(value, isleaf, left_1, left_2, right_1, right_2, pos=pos, grid_size=grid_size)

        return node

    def bfs_for_segmentation(self):
        """

        Returns: segmenta img (np array)

        """
        temp_node = self.root_node
        node_list = [temp_node]

        while len(node_list) > 0:
            temp_node = node_list.pop(0)
            if temp_node.isLeaf is False:
                node_list += [temp_node.left_1, temp_node.left_2, temp_node.right_1, temp_node.right_2]
            else:

                pos_x, pos_y, grid_size = temp_node.pos[0], temp_node.pos[1], temp_node.grid_size
                cv2.rectangle(self.image_grid, (pos_x, pos_y), (pos_x + grid_size, pos_y + grid_size), (0, 255, 255), 1)
        return self.image_grid
        # cv2.imshow("image", self.image_grid)
        # cv2.waitKey(0)  # 防止闪图

    def extract_img_features(self, vector_dim=7) -> list:
        """

        Args:
            min_grid_size:
            vector_dim:

        Returns:
            shape: list
        """
        feature_info = sorted([self.min_grid_size * (2 ** i) for i in range(vector_dim)], reverse=True)  # H -> L
        img_vector = [self.hisogram.get(k, 0) for k in feature_info]
        return img_vector


# image_path = '../greyscale.png'
#
# img = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
#
# s = QuadTreeSolution(img,min_grid_size=4)

# a  = s.bfs_for_segmentation()
# cv2.imshow('1.jpg',a)
# cv2.waitKey()


