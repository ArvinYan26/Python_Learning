from numpy import array
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    def construct(self, grid, all_grid):
        """
        :type grid: List[List[int]]
        :rtype: Node
        """

        root = Node('*', True, None, None, None, None)
        if len(grid) == 1:
            root.isLeaf = True
            root.val = True if grid[0][0] == 1 else False
            all_grid.append(grid)  #将只有一个元素的点加进来
        if self.allValueSame(grid):  # 所有值相等
            root.isLeaf = True
            root.val = True if grid[0][0] == 1 else False
            all_grid.append(grid)  #将所有元素像素值都相等的grid添加进来
        else:  # 并非所有值相等
            halfLength = len(grid) // 2  # 使用 // 表示整除
            root.isLeaf = False  # 如果网格中有值不相等，这个节点就不是叶子节点
            # 使用array来完成二维数组的切片
            root.topLeft, tl_all_grid = self.construct(np.array(grid)[:halfLength, :halfLength], all_grid)
            root.topRight, tr_all_grid = self.construct(np.array(grid)[:halfLength, halfLength:], all_grid)
            root.bottomLeft, bl_all_grid = self.construct(np.array(grid)[halfLength:, :halfLength], all_grid)
            root.bottomRight, br_all_grid = self.construct(np.array(grid)[halfLength:, halfLength:], all_grid)
        return root, all_grid

    def allValueSame(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: boolean
        """
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[0][0] != grid[i][j]:
                    return False
        return True

if __name__ == '__main__':

    img = cv2.imread(r"C:\Users\Yan\Desktop\testdata\COVID-19\1.png", 0)
    plt.imshow(img, "gray")
    plt.show()
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC) #双新兴差值法对图像进行缩放
    plt.imshow(img, "gray")
    plt.show()
    ret, b_img = cv2.threshold(img, 90, 1, cv2.THRESH_BINARY) #1:表示大于阈值后就变为这个数，小于之后就变为0， 二值即0， 1
    #print(b_img)
    plt.imshow(b_img, "gray")
    plt.show()
    S = Solution()
    all_grid = []
    root, all_grid = S.construct(b_img, all_grid)
    #print(all_grid)
    y = []
    count1 = count2 = count3 = count4 = count5 = count6 = count7 = count8 = count9 = count10 = 0
    for i in range(len(all_grid)):
        #print(all_grid[i])
        #print("="*10)
        if len(all_grid[i]) == 512:
            count1 += 1
        if len(all_grid[i]) == 256:
            count2 += 1
        if len(all_grid[i]) == 128:
            count3 += 1
        if len(all_grid[i]) == 64:
            count4 += 1
        if len(all_grid[i]) == 32:
            count5 += 1
        if len(all_grid[i]) == 16:
            count6 += 1
        if len(all_grid[i]) == 8:
            count7 += 1
            #print(count7)
        if len(all_grid[i]) == 4:
            count8 += 1
        if len(all_grid[i]) == 2:
            count9 += 1
        if len(all_grid[i]) == 1:
            count10 += 1

    y.append(count1), y.append(count2), y.append(count3), y.append(count4), y.append(count5), y.append(count6)
    y.append(count7), y.append(count8), y.append(count9), y.append(count10)
    print("y:", len(y), y)
    x = [x for x in range(10)]
    print("x:", len(x), x)

    """
    plt.bar(left=x, height=y, width=0.2, alpha=0.8, color="purple")
    plt.ylim(0, 25000)
    plt.xlabel("Pixel gray value")
    plt.xticks([index + 0.2 for index in x], x)
    plt.ylabel("Number of pixels")
    plt.show()
    """