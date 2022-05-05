import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

start_time = time.time()



normal = []
normal_target = []
viral_pneumonia = []

def convertjpg(path):
    img = cv2.imread(path)
    or_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #转换为matplot的色彩模式
    #print("or_img:", or_img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #转化为灰度图
    #gray_img = cv2.imread(pngfile, cv2.IMREAD_GRAYSCALE) #读取为灰度图,读取的参数必须是路径或者名字，而非一个矩阵图片
    #print(gray_img)
    fft_img = fft(gray_img)

    return or_img, gray_img, fft_img

def fft(img):
    #快速傅里叶变换算法得到频率分布
    f = np.fft.fft2(img)
    #默认结果中心点位置是在左上角,
    #调用fftshift()函数转移到中间位置
    fshift = np.fft.fftshift(f)
    #fft结果是复数, 其绝对值结果是振幅
    fimg = np.log(np.abs(fshift))
    return fimg

def plot_figure(path1, path2):
    # 画图
    or_img1, gray_img1, fft_img1 = convertjpg(path1)
    # print(or_img1)
    plt.figure(1, (12, 12))

    #plt.subplot(231)
    plt.title("Original Img 1")
    plt.imshow(or_img1)

    plt.figure(2, (12, 12))
    #plt.subplot(232)
    plt.title("Gray histigram")
    plt.hist(gray_img1.ravel(), 256, [5, 256])
    plt.xlim((0, 256))
    plt.ylim((0, 30000))
    plt.xlabel("Grayscale")
    plt.ylabel("Number of pixels")

    plt.figure(3, (12, 12))
    #plt.subplot(233)
    plt.title("FFT_Espectro")
    plt.imshow(fft_img1, "gray")

    or_img2, gray_img2, fft_img2 = convertjpg(path2)

    plt.figure(4, (12, 12))
    #plt.subplot(234)
    plt.title("Original Img 2")
    plt.imshow(or_img2)

    plt.figure(5, (12, 12))
    #plt.subplot(235)
    plt.title("Gray histigram")
    plt.hist(gray_img2.ravel(), 256, [5, 256])
    plt.xlim((0, 256))
    plt.ylim((0, 30000))
    plt.xlabel("Grayscale")
    plt.ylabel("Number of pixels")

    plt.figure(6, (12, 12))
    #plt.subplot(236)
    plt.title("FFT_Espectro")
    plt.imshow(fft_img2, "gray")

    plt.show()
"""
count0 = 0
for pngfile in glob.glob(r"C:/Users/Yan/Desktop/test/NORMAL/*.png"):
    normal.append(pngfile)  #此出的pngfile是路径
    count0 += 1
    if count0 == 2:
        break
print(len(normal))
plot_figure(normal[0], normal[1])
"""

path1 = r"C:\Users\Yan\Desktop\newdata\NORMAL\NORMAL (1).png"
path2 = r"C:\Users\Yan\Desktop\newdata\NORMAL\NORMAL (1).png"
plot_figure(path1, path2)

path1 = r"C:\Users\Yan\Desktop\newdata\Viral Pneumonia\Viral Pneumonia (1).png"
path2 = r"C:\Users\Yan\Desktop\newdata\Viral Pneumonia\Viral Pneumonia (2).png"
plot_figure(path1, path2)

path1 = r"C:\Users\Yan\Desktop\newdata\COVID-19\COVID-19 (2).png"
path2 = r"C:\Users\Yan\Desktop\newdata\COVID-19\COVID-19 (3).png"
plot_figure(path1, path2)





"""
#画图
or_img1, gray_img1, fft_img1 = convertjpg(path1)
#print(or_img1)
plt.figure(1, (20, 12))

plt.subplot(231)
plt.title("Original Img 1")
plt.imshow(or_img1)

plt.subplot(232)
plt.title("Gray histigram")
plt.hist(gray_img1.ravel(), 256, [0, 256])
#plt.xlim((0, 256))
#plt.ylim((0, 20000))
plt.xlabel("Grayscale")
plt.ylabel("Number of pixels")

plt.subplot(233)
plt.title("FFT_Espectro")
plt.imshow(fft_img1, "gray")

or_img2, gray_img2, fft_img2 = convertjpg(path2)

plt.subplot(234)
plt.title("Original Img 2")
plt.imshow(or_img2)

plt.subplot(235)
plt.title("Gray histigram")
plt.hist(gray_img2.ravel(), 256, [0, 256])
#plt.xlim((0, 256))
#plt.ylim((0, 20000))
plt.xlabel("Grayscale")
plt.ylabel("Number of pixels")

plt.subplot(236)
plt.title("FFT_Espectro")
plt.imshow(fft_img2, "gray")

plt.show()
"""






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




end_time = time.time()
time = end_time - start_time
print("time:", time)




