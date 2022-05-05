import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

start_time = time.time()
#data = np.empty(shape=[0, 2916], dtype=int)


#存储处理后的图像
normal = []
viral_pneumonia = []
covid_19 = []

min_value, max_value, steps = -40, 160, 10
Threshold = [x for x in range(min_value, max_value, steps)]

def whole_historgram(data, count):
    his_img = []
    plt.figure()
    for i in range(len(data)):
        if i % (len(data)/count) == 0: #找到原灰度图像，画出直方图
            his_img.append(data[i])
    #print(len(his_img), data[0])
    for i in range(len(his_img)):
        #print(i, his_img[i])
        #if i == 3:
            #print(his_img[i].ravel())
        plt.hist(his_img[i].ravel(), 256, [0, 256])
        plt.subplot(count/2, 2, i + 1)
        plt.xlabel("Pixel gray value")
        plt.ylabel("Number of pixels")
    plt.show()

def binary_image(image, data):
    for i in Threshold:
        #img1 = []
        #img1.append(image1)
        #for i in Threshold:
        #threshold = 120       #ˈθreʃhoʊld
        #print("i:", i)
        ret, b_img = cv2.threshold(image, i, 255, cv2.THRESH_BINARY)
        data.append(b_img)

    return data
def convertjpg(pngfile, data):

    img = cv2.imread(pngfile, cv2.IMREAD_GRAYSCALE) #读取为灰度图,
    #因为opencv读取图片方式和matplot不一样，所以显示结果不同，所以不再显示原图，直接显示灰度图和二值图像
    #img = cv2.imread(pngfile) #读取为灰度图,
    #data.append(img) #添加原图
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #data.append(img)
    #图像去均值
    img1 = img.reshape(1, -1) #图像变为一个向量
    mean_image = np.mean(img1[0])  #求像素均值
    #print("mean_image:", mean_image)
    #new_img = (img1 - mean_image).reshape(img.shape[0], img.shape[1])
    new_img = (img - mean_image) + 60 #矩阵减去一个数，就是矩阵中的每一个元素减去这个数，真棒，哈哈
    data.append(new_img)
    #print("img:", img)
    #print("new_img:", new_img)
    #new_data = binary_image(new_img, data)

    return data

def draw_graph(data, count):
    plt.figure()
    for i in range(len(data)):
        #print("i:", i)
        #plt.annotate('covid-19', xy=(0, 0), xytext=(1, 1))
        plt.subplot(count, len(data)/count0, i+1)
        plt.imshow(data[i], "gray") #这是因为我们还是直接使用plt显示图像，它默认使用三通道显示图像，需要我们在可视化函数里多指定一个参数
        plt.xticks([])
        plt.yticks([])
    plt.show()




def fractal_dimension(Z, threshold):
    # Only for 2d image
    assert(len(Z.shape) == 2) #assert ： 判断语句，如果是二维图像就执行，不是的话，直接报错，不再执行

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        #axis=0, 按列计算， 1：按行计算
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    # Transform Z into a binary array
    Z = (Z < threshold)
    #plt.draw()
    #plt.show()

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def get_line_chart_data(data, count):
    #gray_img = []
    #for n in range(len(data)):
        #if n % (len(data) / count) == 0:
            #gray_img.append(data[n])

    l = []
    for j in range(len(data)):
        x = []
        for i in Threshold:
            y = fractal_dimension(data[j], i)
            x.append(y)
        l.append(x)

    return l
#读取图片数据，转化为矩阵
count0 = 0
for pngfile in glob.glob(r"C:/Users/Yan/Desktop/screen shot/NORMAL/*.png"):
    normal = convertjpg(pngfile, normal)
    count0 += 1
    if count0 == 4:
        break
print(len(normal))


count1 = 0
for pngfile in glob.glob(r"C:/Users/Yan/Desktop/screen shot/Viral Pneumonia/*.png"):
    viral_pneumonia = convertjpg(pngfile, viral_pneumonia)
    count1 += 1
    if count1 == 4:
        break
print(len(viral_pneumonia))

count2 = 0
for pngfile in glob.glob(r"C:/Users/Yan/Desktop/screen shot/COVID-19/*.png"):
    covid_19 = convertjpg(pngfile, covid_19)
    count2 += 1
    if count2 == 4:
        break
print(len(covid_19))

"""
#画灰度像素直方图
whole_historgram(normal, count0)
whole_historgram(viral_pneumonia, count1)
whole_historgram(covid_19, count2)

#显示不同阈值的二值图像
draw_graph(normal, count0)
draw_graph(viral_pneumonia, count1)
draw_graph(covid_19, count2)
"""

#获取维度折线图数据
D = get_line_chart_data(normal, count0)
D1 = get_line_chart_data(viral_pneumonia, count1)
D2 = get_line_chart_data(covid_19, count2)

#画维度折线图
plt.figure("Fractal Demension")

plt.plot(Threshold, D[0], color="#00ff00", label="normal0")
plt.plot(Threshold, D[1], color="#00ff5f", label="normal1")
plt.plot(Threshold, D[2], color="#00ff87", label="normal2")
plt.plot(Threshold, D[3], color="#00ffaf", label="normal3")

plt.plot(Threshold, D1[0], color="#ff0000", label="Viral Pneumonia0")
plt.plot(Threshold, D1[1], color="#ff005f", label="Viral Pneumonia1")
plt.plot(Threshold, D1[2], color="#ff0087", label="Viral Pneumonia2")
plt.plot(Threshold, D1[3], color="#ff00af", label="Viral Pneumonia3")

plt.plot(Threshold, D2[0], color="#af00ff", label="covid0")
plt.plot(Threshold, D2[1], color="#af5fff", label="covid1")
plt.plot(Threshold, D2[2], color="#af87ff", label="covid2")
plt.plot(Threshold, D2[3], color="#afafff", label="covid3")

plt.legend()
plt.xlabel("Thershold")
plt.ylabel("Desmension")
plt.show()
"""
data0 = np.array(data0)
data1 = np.array(data1)
data2 = np.array(data2)
data = np.vstack((data0, data1, data2))
data_target = data_target0 + data_target1 + data_target2
end_time = time.time()
time = end_time - start_time
print("time:", time)
"""


