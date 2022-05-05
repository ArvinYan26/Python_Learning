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
    def __init__(self):
        self.all_grid = []

    def construct(self, grid, stddv_threshold):
        """
        :type grid: List[List[int]]
        :rtype: Node
        """
        mean, stddv = cv2.meanStdDev(grid)
        root = Node('*', True, None, None, None, None)
        """
        if len(grid) == 1:
            root.isLeaf = True
            root.val = True if stddv <= 0.5 else False
            all_grid.append(grid)  #将只有一个元素的点加进来
        """

        if stddv <= stddv_threshold:  # 均方差等于零，那么就代表已经分到单个像素级别了，或者stddv <= 0.8, 分到2*2，就不再划分
            root.isLeaf = True
            root.val = True     #或者是直接等于零，分到一个像素位置
            self.all_grid.append(grid)  #将所有元素像素值都相等的grid添加进来
        else:  # 并非所有值相等
            halfLength = len(grid) // 2  # 使用 // 表示整除
            root.isLeaf = False  # 如果网格中有值不相等，这个节点就不是叶子节点
            # 使用array来完成二维数组的切片
            root.topLeft, tl_all_grid = self.construct(np.array(grid)[:halfLength, :halfLength], stddv_threshold)
            root.topRight, tr_all_grid = self.construct(np.array(grid)[:halfLength, halfLength:], stddv_threshold)
            root.bottomLeft, bl_all_grid = self.construct(np.array(grid)[halfLength:, :halfLength], stddv_threshold)
            root.bottomRight, br_all_grid = self.construct(np.array(grid)[halfLength:, halfLength:], stddv_threshold)
        #return root
        return root, self.all_grid

    def count_blocks(self, grid, stddv_threshold):
        root, all_grid = self.construct(grid, stddv_threshold)
        y = []
        count1 = count2 = count3 = count4 = count5 = count6 = count7 = count8 = count9 = count10 = 0
        for i in range(len(all_grid)):
            # print(all_grid[i])
            # print("="*10)
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
                # print(count7)
            if len(all_grid[i]) == 4:
                count8 += 1
            if len(all_grid[i]) == 2:
                count9 += 1
            if len(all_grid[i]) == 1:
                count10 += 1
        y.append(count1), y.append(count2), y.append(count3), y.append(count4), y.append(count5), y.append(count6)
        y.append(count7), y.append(count8), y.append(count9), y.append(count10)
        return y



if __name__ == '__main__':

    img = cv2.imread(r"C:\Users\Yan\Desktop\c.png", 0)
    img1 = cv2.imread(r"C:\Users\Yan\Desktop\b.png", 0)
    #img = cv2.imread(r"C:\Users\Yan\Desktop\128.png", 0)
    plt.imshow(img, "gray")
    plt.show()
    #img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC) #双新兴差值法对图像进行缩放
    #img1 = cv2.resize(img1, (512, 512), interpolation=cv2.INTER_CUBIC) #双新兴差值法对图像进行缩放

    plt.imshow(img, "gray")
    plt.show()
    var4 = np.var(img1)
    var2 = np.var(img)
    print(var4, var4)

    """
    mean, stddv = cv2.meanStdDev(img)
    print("stddv:", stddv)
    mean1, stddv1 = cv2.meanStdDev(img1)
    print("stddv1:", stddv1)
    S = Solution()
    #all_grid = []
    #all_grid1 = []
    every_blocks_num = S.count_blocks(img)
    print(every_blocks_num)
    """


    """
    root, all_grid = S.construct(img)
    root1, all_grid1 = S.construct(img1)
    #print(all_grid)
    all_grid = all_grid
    all_grid1 = all_grid1
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

    y1 = []
    count1 = count2 = count3 = count4 = count5 = count6 = count7 = count8 = count9 = count10 = 0
    for i in range(len(all_grid)):
        # print(all_grid[i])
        # print("="*10)
        if len(all_grid1[i]) == 512:
            count1 += 1
        if len(all_grid1[i]) == 256:
            count2 += 1
        if len(all_grid1[i]) == 128:
            count3 += 1
        if len(all_grid1[i]) == 64:
            count4 += 1
        if len(all_grid1[i]) == 32:
            count5 += 1
        if len(all_grid1[i]) == 16:
            count6 += 1
        if len(all_grid1[i]) == 8:
            count7 += 1
            # print(count7)
        if len(all_grid1[i]) == 4:
            count8 += 1
        if len(all_grid1[i]) == 2:
            count9 += 1
        if len(all_grid1[i]) == 1:
            count10 += 1

    y1.append(count1), y1.append(count2), y1.append(count3), y1.append(count4), y1.append(count5), y1.append(count6)
    y1.append(count7), y1.append(count8), y1.append(count9), y1.append(count10)
    print("y1:", len(y1), y1)
    x = [x for x in range(10)]
    print("x:", len(x), x)
    """
    """
    plt.bar(left=x, height=y, width=0.2, alpha=0.8, color="purple")
    plt.ylim(0, 25000)
    plt.xlabel("Pixel gray value")
    plt.xticks([index + 0.2 for index in x], x)
    plt.ylabel("Number of pixels")
    plt.show()
    """