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
mean_data = []    #存储每张图片的均值

th_value = []   #存储每一个图像的阈值范围

#min_value, max_value, steps = 72, 93, 3
#Threshold = [x for x in range(min_value, max_value, steps)]

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

def binary_image(image, data, Threshold):
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
    data.append(img) #存储的原始灰度图像
    #获取每张原灰度图像的均值
    (mean, stddv) = cv2.meanStdDev(img)  ##计算图像的均值和方差
    mean_data.append(mean[0][0]) #将每张灰度图的均值存储起来

    mean_value = int(mean[0][0])    #灰度图像均值
    #设置二值图像与之参数
    steps = 5  #阈值变化步长
    plt_image_num = 6 #每张灰度图显示的二值图像个数
    min_value, max_value, = int(mean_value - steps*plt_image_num/2), int(mean_value + steps*plt_image_num/2)   #3*266张照片
    Threshold = [x for x in range(min_value, max_value, steps)]
    th_value.append(Threshold)
    #获取二值图像
    new_data = binary_image(img, data, Threshold)  #在原来的data里面添加二值图像，返回new_data

    return new_data

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

def get_line_chart_data(data, count, Threshold):
    gray_img = []
    for n in range(len(data)):
        if n % (len(data) / count) == 0:
            gray_img.append(data[n])
    #print(len(Threshold))
    l = []
    for j in range(len(gray_img)):
        #print(len(gray_img), len(Threshold[j]))
        x = []
        for i in Threshold[j]:
            y = fractal_dimension(gray_img[j], i)
            x.append(y)
        l.append(x)

    return l
#读取图片数据，转化为矩阵
count0 = 0
for pngfile in glob.glob(r"C:/Users/Yan/Desktop//COVID-19-c/NORMAL/*.png"):
    normal = convertjpg(pngfile, normal)  #存储的处理后的原图像和二值图像
    count0 += 1
    if count0 == 4:
        break
print(len(normal))


count1 = 0
for pngfile in glob.glob(r"C:/Users/Yan/Desktop//COVID-19-c/Viral Pneumonia/*.png"):
    viral_pneumonia = convertjpg(pngfile, viral_pneumonia)
    count1 += 1
    if count1 == 4:
        break
print(len(viral_pneumonia))

count2 = 0
for pngfile in glob.glob(r"C:/Users/Yan/Desktop//COVID-19-c/COVID-19/*.png"):
    covid_19 = convertjpg(pngfile, covid_19)
    count2 += 1
    if count2 == 4:
        break
print(len(covid_19))

#print(len(mean_data), mean_data)

#画灰度像素直方图
whole_historgram(normal, count0)
whole_historgram(viral_pneumonia, count1)
whole_historgram(covid_19, count2)

#显示不同阈值的二值图像
draw_graph(normal, count0)
draw_graph(viral_pneumonia, count1)
draw_graph(covid_19, count2)

#print(len(normal), len(viral_pneumonia), len(covid_19))
#获取维度折线图数据
#print(np.array(th_value))
print("mean_data:", mean_data)
th0 = th_value[:4]
print("th0:", len(th0), th0)
D = get_line_chart_data(normal, count0, th0)

th1 = th_value[4:8]
print("th1:", th1)
D1 = get_line_chart_data(viral_pneumonia, count1, th1)

th2 = th_value[8:]
print("th2:", th2)
D2 = get_line_chart_data(covid_19, count2, th2)

#画维度折线图
plt.figure("Fractal Demension")

plt.plot(th0[0], D[0], color="#00ff00", label="normal0")
plt.plot(th0[1], D[1], color="#00ff5f", label="normal1")
plt.plot(th0[2], D[2], color="#00ff87", label="normal2")
plt.plot(th0[3], D[3], color="#00ffaf", label="normal3")

plt.plot(th1[0], D1[0], color="#ff0000", label="Viral Pneumonia0")
plt.plot(th1[1], D1[1], color="#ff005f", label="Viral Pneumonia1")
plt.plot(th1[2], D1[2], color="#ff0087", label="Viral Pneumonia2")
plt.plot(th1[3], D1[3], color="#ff00af", label="Viral Pneumonia3")

plt.plot(th2[0], D2[0], color="#af00ff", label="covid0")
plt.plot(th2[1], D2[1], color="#af5fff", label="covid1")
plt.plot(th2[2], D2[2], color="#af87ff", label="covid2")
plt.plot(th2[3], D2[3], color="#afafff", label="covid3")

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


