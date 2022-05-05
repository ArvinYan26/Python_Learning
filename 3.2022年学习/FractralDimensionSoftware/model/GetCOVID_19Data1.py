import glob
import cv2
import numpy as np
import time

class GetData():
    def __init__(self, max_value, min_value, steps):
        self.min_value, self.max_value, steps = max_value, min_value, steps
        self.Threshold = [x for x in range(self.min_value, self.max_value, steps)]
        self.grid_rank = None

    def fd_feature(self, pngfile, class_num, data, data_target):
        img = cv2.imread(pngfile, cv2.IMREAD_GRAYSCALE) #读取为灰度图,
        x = []
        for i in self.Threshold:
            ret, b_img = cv2.threshold(img, i, 255, cv2.THRESH_BINARY)
            d = self.fractal_dimension(b_img, i)
            x.append(d)
        x = np.array(x)
        data.append(x)
        data_target.append(class_num)

    def fractal_dimension(self, Z, threshold):
        # Only for 2d image
        assert(len(Z.shape) == 2) #assert ： 判断语句，如果是二维图像就执行，不是的话，直接报错，不再执行
        def boxcount(Z, k):
            #axis=0, 按列计算， 1：按行计算
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                   np.arange(0, Z.shape[1], k), axis=1)

            return len(np.where((S > 0) & (S < k*k))[0])
        Z = (Z < threshold)
        p = min(Z.shape)
        n = 2**np.floor(np.log(p)/np.log(2))
        n = int(np.log(n)/np.log(2))
        sizes = 2**np.arange(n, 1, -1)
        counts = []
        for size in sizes:
            counts.append(boxcount(Z, size))
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]

    def get_data(self, path0, path1):
        """"
        percent:训练集比例
        """
        data0 = []
        data2 = []
        data_target0 = []
        data_target2 = []
        #读取图片数据，转化为矩阵,不同的特征提取函数，不同的列明，所以特征提取函数也需要改
        count0 = 0

        for pngfile in glob.glob(path0):
            self.fd_feature(pngfile, 0, data0, data_target0)    #不同的特征提取方法函数不同，注意修改
            count0 += 1
            if count0 == 150:
                break
        count1 = 0

        for pngfile in glob.glob(path1):
            self.fd_feature(pngfile, 1, data2, data_target2)
            count1 += 1
            if count1 == 150:
                break

        data0 = np.array(data0)
        data2 = np.array(data2)
        data = np.vstack((data0, data2))
        # 合并类别号
        data_target = np.array(data_target0 + data_target2)

        return data, data_target





