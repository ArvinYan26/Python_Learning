import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


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

#读取图像
img = cv.imread(r"C:/Users/Yan/Desktop/COVID-19 (7).png", 0)

#快速傅里叶变换算法得到频率分布
f = np.fft.fft2(img)
#plt.imshow(img, 'gray'), plt.title("ffts")
#plt.show()

#默认结果中心点位置是在左上角,
#调用fftshift()函数转移到中间位置
fshift = np.fft.fftshift(f)

#fft结果是复数, 其绝对值结果是振幅
fimg = np.log(np.abs(fshift))
print("fimg:", fimg.shape, fimg)
#展示结果
plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Fourier')
plt.axis('off')
plt.subplot(122), plt.imshow(fimg, 'gray'), plt.title('Fourier Fourier')
plt.axis('off')
plt.show()

plt.hist(fimg.ravel(), 10, [0, 10])
#plt.subplot(count/2, 2, i + 1)
plt.xlabel("Pixel gray value")
plt.ylabel("Number of pixels")
plt.show()


