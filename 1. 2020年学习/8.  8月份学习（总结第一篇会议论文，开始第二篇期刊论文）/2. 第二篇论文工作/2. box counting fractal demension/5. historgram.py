import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import numpy as np

def whole_historgram(img):
    plt.hist(img.ravel(), 256, [0, 256])
    plt.xlabel("Pixel gray value")
    plt.ylabel("Number of pixels")
    plt.show()


#image = cv2.imread(r"C:\Users\Yan\Desktop\xiaoqiao.jfif", cv2.IMREAD_GRAYSCALE) #读取灰度图
image = cv2.imread(r"C:\Users\Yan\Desktop\testdata\COVID-19\4.png", 0)
image1 = cv2.imread(r"C:\Users\Yan\Desktop\testdata\COVID-19\6.png", 0)
#plt original historgram
plt.title("image")
whole_historgram(image)

plt.title("image1")
whole_historgram(image1)
#二值化图像, 127：阈值， 255：图像最大像素值， cv2.THRESH_BINARY:二值化阈值方法，
#ret:返回的阈值， thresh1:返回的二值图像
img = []
img1 = []
img.append(image)
img1.append(image1)
min_value, max_value, steps = 60, 180, 20
Threshold = [x for x in range(min_value, max_value, steps)]
for i in Threshold:
    #img1 = []
    #img1.append(image1)
    #for i in Threshold:
    #threshold = 120       #ˈθreʃhoʊld
    #print("i:", i)
    ret, b_img = cv2.threshold(image, i, 255, cv2.THRESH_BINARY)
    ret1, b_img1 = cv2.threshold(image1, i, 255, cv2.THRESH_BINARY)
    img.append(b_img)
    img1.append(b_img1)
    #print(ret, thresh1)
    #th3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    #fig = plt.figure()

    #print(type(img))

plt.figure()
for i in range(len(img)):
    #print("i:", i)
    #plt.annotate('covid-19', xy=(0, 0), xytext=(1, 1))
    plt.subplot(2, len(img), i+1)
    plt.imshow(img1[i], "gray") #这是因为我们还是直接使用plt显示图像，它默认使用三通道显示图像，需要我们在可视化函数里多指定一个参数
    plt.xticks([])
    plt.yticks([])

    #plt.annotate('covid-19_short', xy=(0, 0), xytext=(1, 1))
    plt.subplot(2, 7, len(img) + i + 1)
    plt.imshow(img[i], "gray")

    plt.xticks([])
    plt.yticks([])
plt.show()



