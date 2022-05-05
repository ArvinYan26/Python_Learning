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
normal_target = []
viral_pneumonia = []
viral_pneumonia_target = []

covid_19 = []
covid_19_target = []

min_value, max_value, steps = 10, 240, 10   #90-160
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
        plt.subplot(count, 1, i + 1)
        plt.xlabel("Pixel gray value")
        plt.ylabel("Number of pixels")
    plt.show()


def convertjpg(pngfile, data):
    img = cv2.imread(pngfile)
    data.append(img)    #原图像

    img = cv2.imread(pngfile, cv2.IMREAD_GRAYSCALE) #读取为灰度图,
    #data.append(img)   #灰度图
    img = fft(img)
    data.append(img)   #傅里叶变换后的图像

    return data

def fft(img):
    #快速傅里叶变换算法得到频率分布
    f = np.fft.fft2(img)

    #默认结果中心点位置是在左上角,
    #调用fftshift()函数转移到中间位置
    fshift = np.fft.fftshift(f)

    #fft结果是复数, 其绝对值结果是振幅
    fimg = np.log(np.abs(fshift))
    return fimg

def draw_graph(data, count):
    plt.figure()
    for i in range(len(data)):
        #print("i:", i)d
        """
        if i % 2 == 0:
            print(i)
            print(data[i])
            img = cv2.imread(data[i], cv2.IMREAD_GRAYSCALE) #灰度图
            plt.hist(img.ravel(), 256, [0, 256])
        """

        plt.subplot(count, len(data)/count0, i+1)
        print(i, data[i].shape)
        plt.imshow(data[i], "gray") #这是因为我们还是直接使用plt显示图像，它默认使用三通道显示图像，需要我们在可视化函数里多指定一个参数
        plt.xticks([])
        plt.yticks([])
    plt.show()

"""
def get_line_chart_data(data):
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
"""

#读取图片数据，转化为矩阵
count0 = 0
for pngfile in glob.glob(r"C:/Users/Yan/Desktop/COVID-19-c/NORMAL/*.png"):
    normal = convertjpg(pngfile, normal)
    count0 += 1

    if count0 == 3:
        break
#print(len(normal))


count1 = 0
for pngfile in glob.glob(r"C:/Users/Yan/Desktop/COVID-19-c/Viral Pneumonia/*.png"):
    viral_pneumonia = convertjpg(pngfile, viral_pneumonia)
    count1 += 1
    if count1 == 3:
        break
#print(len(viral_pneumonia))

count2 = 0
for pngfile in glob.glob(r"C:/Users/Yan/Desktop/COVID-19-c/COVID-19/*.png"):
    covid_19 = convertjpg(pngfile, covid_19)
    count2 += 1
    if count2 == 3:
        break
print(len(covid_19))

#print(normal, viral_pneumonia, covid_19)

"""
#画灰度像素直方图
whole_historgram(normal, count0)
whole_historgram(viral_pneumonia, count1)
whole_historgram(covid_19, count2)
"""

#显示不同阈值的二值图像
draw_graph(normal, count0)
draw_graph(viral_pneumonia, count1)
draw_graph(covid_19, count2)


"""
#获取维度折线图数据
D = get_line_chart_data(normal)
D1 = get_line_chart_data(viral_pneumonia)
D2 = get_line_chart_data(covid_19)

print(len(np.array(D)), len(np.array(D1)), len(np.array(D2)))
"""

"""
#画维度折线图
plt.figure("Fractal Demension")

plt.plot(Threshold, D[0], color="#00ff00", label="normal0")
plt.plot(Threshold, D[1], color="#00ff5f", label="normal1")
plt.plot(Threshold, D[2], color="#00ff87", label="normal2")
plt.plot(Threshold, D[3], color="#00ffaf", label="normal3")
plt.plot(Threshold, D[4], color="#00ffaf", label="normal4")
plt.plot(Threshold, D[5], color="#00ffaf", label="normal5")
plt.plot(Threshold, D[6], color="#00ffaf", label="normal5")

plt.plot(Threshold, D1[0], color="#ff0000", label="Viral0")
plt.plot(Threshold, D1[1], color="#ff005f", label="Viral1")
plt.plot(Threshold, D1[2], color="#ff0087", label="Viral2")
plt.plot(Threshold, D1[3], color="#ff00af", label="Viral3")
plt.plot(Threshold, D1[4], color="#ff00af", label="Viral4")
plt.plot(Threshold, D1[5], color="#ff00af", label="Viral5")
plt.plot(Threshold, D1[6], color="#ff00af", label="Viral5")

plt.plot(Threshold, D2[0], color="#af00ff", label="covid0")
plt.plot(Threshold, D2[1], color="#af5fff", label="covid1")
plt.plot(Threshold, D2[2], color="#af87ff", label="covid2")
plt.plot(Threshold, D2[3], color="#afafff", label="covid3")
plt.plot(Threshold, D2[4], color="#afafff", label="covid4")
plt.plot(Threshold, D2[5], color="#afafff", label="covid5")
plt.plot(Threshold, D2[6], color="#afafff", label="covid5")

plt.legend()
plt.xlabel("Thershold")
plt.ylabel("Desmension")
plt.show()
"""

"""
data0 = np.array(D)
#print(data0)
data1 = np.array(D1)
data2 = np.array(D2)
data = np.vstack((data0, data1, data2))
#print(data)
#data_target = normal_target + viral_pneumonia_target + covid_19_target
#print(data)
print(data.shape)
#save = pd.DataFrame(data)
#save.to_csv(r"C:/Users/Yan/Desktop/fractal_demension_145_175_5.csv", index=False, header=True)
"""

end_time = time.time()
time = end_time - start_time
print("time:", time)




