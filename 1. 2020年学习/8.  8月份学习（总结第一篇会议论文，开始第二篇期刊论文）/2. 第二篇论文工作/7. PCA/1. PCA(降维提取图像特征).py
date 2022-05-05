from sklearn.decomposition import PCA
import glob
import cv2
import numpy as np
import pandas as pd
import time

start_time = time.time()
def convertjpg(pngfile, data, width=256, height=256):
    img = cv2.imread(pngfile, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图,
    #dst = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)  # interpola：双线性插值方式
    #dst = dst.reshape(1, -1)
    img = img.reshape(1, -1)
    data.append(img[0])
    #data.append(dst[0])
    return data

def get_data():
    data0 = []
    data1 = []
    data2 = []

    # 读取图片数据，转化为矩阵
    count0 = 0
    for pngfile in glob.glob(r"C:/Users/Yan/Desktop/COVID-19-c/NORMAL/*.png"):
        data0 = convertjpg(pngfile, data0)
        count0 += 1
        if count0 == 150:
            break

    count = 0
    for pngfile in glob.glob(r"C:/Users/Yan/Desktop/COVID-19-c/COVID-19/*.png"):
        data1 = convertjpg(pngfile, data1)
        count += 1
        if count == 150:
            break

    count1 = 0
    for pngfile in glob.glob(r"C:/Users/Yan/Desktop/COVID-19-c/Viral Pneumonia/*.png"):
        data2 = convertjpg(pngfile, data2)
        count1 += 1
        if count1 == 150:
            break

    data0 = np.array(data0)
    data1 = np.array(data1)
    data2 = np.array(data2)
    data = np.vstack((data0, data1, data2))
    print(data.shape, data[0].shape)
    pca = PCA(n_components=0.99)  ###PCA，提取特征为256
    p = pca.fit_transform(data)  ###对原始数据进行降维
    #print(p.shape, p)
    #save = pd.DataFrame(data0)
    #save.to_csv(r"C:/Users/Yan/Desktop/pca0.95.csv", index=False, header=True)


    data = pd.DataFrame(p)
    data.to_csv(r"C:/Users/Yan/Desktop/pca0.99.csv", index=False, header=True)

    return p


#end_time = time.time()
#time = end_time - start_time
# print("time:", time)

data = get_data()
print(data.shape, data[0].shape)



