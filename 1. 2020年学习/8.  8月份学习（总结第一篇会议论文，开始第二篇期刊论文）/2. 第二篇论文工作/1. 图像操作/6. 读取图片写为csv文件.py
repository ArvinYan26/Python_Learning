import os.path
import glob
import cv2
import numpy as np
import pandas as pd
import time

start_time = time.time()
data = []
data_target = []


def convertjpg(pngfile, path, width=128, height=128):
    img = cv2.imread(pngfile, cv2.COLOR_BGR2GRAY)  # 读取为灰度图
    dst = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)  # interpola：双线性插值方式
    # cv2.imwrite(os.path.join(outdir, os.path.basename(pngfile)), dst)
    dst = dst.reshape(1, -1)
    dst = list(dst)
    # save = pd.DataFrame(data)
    # save.to_csv(path, index=False, header=True)

    # dst = dst[0]
    # print(dst)
    data.append(dst)
    # data_target.append(class_num)


for pngfile in glob.glob(r"C:/Users/Yan/Desktop/COVID-19/NORMAL/*.png"):
    convertjpg(pngfile, r'C:/Users/Yan/Desktop/data/NORMAL.csv')

# data = np.array(data)
# print(data.shape, len(data_target))

# print(data, data_target)


for pngfile in glob.glob(r"C:/Users/Yan/Desktop/COVID-19/Viral Pneumonia/*.png"):
    convertjpg(pngfile, r'C:\Users\Yan\Desktop\data\Viral Pneumonia.csv', 1)

# print(np.array(data).shape, len(data_target))

for pngfile in glob.glob(r"C:/Users/Yan/Desktop/COVID-19/COVID-19/*.png"):
    convertjpg(pngfile, r'C:\Users\Yan\Desktop\data\COVID-19.csv', 2)


print(data)
data = np.array(data)
print(len(data), len(data_target))
print(data, data_target)

"""
for i in range(len(data)):
    data = data[i]
    save = pd.DataFrame(data)
    save.to_csv(r"C:/Users/Yan/Desktop/data.csv", index=False, header=True)
"""

#save = pd.DataFrame(data)
#save.to_csv(r"C:/Users/Yan/Desktop/data.csv", index=False, header=True)

end_time = time.time()
time = end_time - start_time
print("time:", time)

