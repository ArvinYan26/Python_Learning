import pyqtree
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread(r"C:\Users\Yan\Desktop\COVID-19\COVID-19\COVID-19 (2).png", 0)  # 读取灰度图
img1 = cv2.imread(r"C:\Users\Yan\Desktop\resize data\COVID-19\COVID-19 (2).png", 0) #读取灰度图
# print(img1)
plt.subplot(2, 2, 1)
plt.imshow(img, "gray")
# plt.subplot(2, 2, 1)
# plt.imshow(img1, "gray")


# 二值化图像
img2 = img.reshape(1, -1)  # 图像变为一个向量
mean_image = np.mean(img2[0])  # 求像素均值
print(mean_image)
# new_img = (img - mean_image)  # 矩阵减去一个数，就是矩阵中的每一个元素减去这个数，真棒，哈哈
ret, b_img = cv2.threshold(img, mean_image, 255, cv2.THRESH_BINARY)
plt.subplot(222)
plt.imshow(b_img, "gray")


# Definition for a QuadTree node.
class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight



class Solution:
    def construct(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: Node
        """
        isLeaf = self.isQuadTree(grid)
        _len = len(grid)
        if isLeaf == None:
            mid = _len // 2
            topLeftGrid = [[grid[i][j] for j in range(mid)] for i in range(mid)]
            topRightGrid = [[grid[i][j] for j in range(mid, _len)] for i in range(mid)]
            bottomLeftGrid = [[grid[i][j] for j in range(mid)] for i in range(mid, _len)]
            bottomRightGrid = [[grid[i][j] for j in range(mid, _len)] for i in range(mid, _len)]
            node = Node(True, False, self.construct(topLeftGrid), self.construct(topRightGrid),
                        self.construct(bottomLeftGrid), self.construct(bottomRightGrid))
        elif isLeaf == False:
            node = Node(False, True, None, None, None, None)
        else:
            node = Node(True, True, None, None, None, None)
        return node

    def isQuadTree(self, grid):
        _len = len(grid)
        _sum = 0
        for i in range(_len):
            _sum += sum(grid[i])
        if _sum == _len ** 2:
            return True
        elif _sum == 0:
            return False
        else:
            return None

if __name__ == '__main__':
    print(img1.shape)
    a = [[1, 1, 1, 1, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 0, 0]]
    s = Solution()
    node = s.construct(a)

    print(node)

