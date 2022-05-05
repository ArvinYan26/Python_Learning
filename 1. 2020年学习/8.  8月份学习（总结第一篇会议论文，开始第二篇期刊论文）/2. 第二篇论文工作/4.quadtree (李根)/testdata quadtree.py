import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from quadtree_historgram import Solution

start_time = time.time()
#data = np.empty(shape=[0, 2916], dtype=int)


#存储处理后的图像
normal = []
viral_pneumonia = []
covid_19 = []
#小于这个值，就不再分割,
stddv_threshold = 20  #小于这个值，就不再分割

def convertjpg(pngfile, data, stddv_threshold):

    img = cv2.imread(pngfile, 0) #0:读取为灰度图,
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)  # 双新兴差值法对图像进行缩放
    #data.append(img) #存储的缩放后的灰度图像
    S = Solution()
    every_blocks_num = S.count_blocks(img, stddv_threshold)
    data.append(every_blocks_num)

    return data

#读取图片数据，转化为矩阵
count0 = 0
for pngfile in glob.glob(r"C:/Users/Yan/Desktop/COVID-19-c/NORMAL/*.png"):
    normal = convertjpg(pngfile, normal, stddv_threshold)  #存储的处理后的原图像和二值图像
    count0 += 1
    if count0 == 4:
        break
print(len(normal), np.array(normal))


count1 = 0
for pngfile in glob.glob(r"C:/Users/Yan/Desktop/COVID-19-c/Viral Pneumonia/*.png"):
    viral_pneumonia = convertjpg(pngfile, viral_pneumonia, stddv_threshold)
    count1 += 1
    if count1 == 4:
        break
print(len(viral_pneumonia), np.array(viral_pneumonia))

count2 = 0
for pngfile in glob.glob(r"C:/Users/Yan/Desktop/COVID-19-c/COVID-19/*.png"):
    covid_19 = convertjpg(pngfile, covid_19, stddv_threshold)
    count2 += 1
    if count2 == 4:
        break
print(len(covid_19), np.array(covid_19))