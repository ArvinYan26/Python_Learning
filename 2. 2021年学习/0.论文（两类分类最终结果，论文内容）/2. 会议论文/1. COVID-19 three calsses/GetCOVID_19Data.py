import os.path
import glob
import cv2
import numpy as np
import pandas as pd
import time

start_time = time.time()
def convertjpg(pngfile, class_num, data, data_target, width=256, height=256):
    img = cv2.imread(pngfile, cv2.IMREAD_GRAYSCALE) #读取为灰度图,
    dst = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)   #interpola：双线性插值方式
    dst = dst.reshape(1, -1)
    data.append(dst[0])
    data_target.append(class_num)

def get_data():
    data0 = []
    data1 = []
    data2 = []
    data_target0 = []
    data_target1 = []
    data_target2 = []

    #读取图片数据，转化为矩阵
    count0 = 0
    for pngfile in glob.glob(r"C:/Users/Yan/Desktop/COVID-19-c/NORMAL/*.png"):
        convertjpg(pngfile, 0, data0, data_target0)
        count0 += 1
        if count0 == 150:
            break

    count = 0
    for pngfile in glob.glob(r"C:/Users/Yan/Desktop/COVID-19-c/COVID-19/*.png"):
        convertjpg(pngfile, 1, data1, data_target1)
        count += 1
        if count == 150:
            break

    count1 = 0
    for pngfile in glob.glob(r"C:/Users/Yan/Desktop/COVID-19-c/Viral Pneumonia/*.png"):
        convertjpg(pngfile, 2, data2, data_target2)
        count1 += 1
        if count1 == 150:
            break

    data0 = np.array(data0)
    data1 = np.array(data1)
    data2 = np.array(data2)

    """
    print(data0, data_target0)
    print("data1:", data1, data_target1)
    print(data2, data_target2)
    """

    data = np.vstack((data0, data1, data2))
    data_target = data_target0 + data_target1 + data_target2

    #data = np.vstack((data0, data1))
    #data_target = data_target0 + data_target1

    #print(data.shape, len(data_target))
    #print(data, data_target)

    """
    save = pd.DataFrame(data0)
    save.to_csv(r"C:/Users/Yan/Desktop/data.csv", index=False, header=True)
    
    data_target = pd.DataFrame(data_target)
    data_target.to_csv(r"C:/Users/Yan/Desktop/data_taget.csv", index=False, header=True)
    """
    return data, data_target


end_time = time.time()
time = end_time - start_time
#print("time:", time)



