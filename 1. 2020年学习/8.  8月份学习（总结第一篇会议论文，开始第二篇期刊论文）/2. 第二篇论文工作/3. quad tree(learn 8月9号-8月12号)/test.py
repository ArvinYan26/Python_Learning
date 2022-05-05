import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread(r"C:\Users\Yan\Desktop\testdata\COVID-19\1.png", 0)
img1 = cv2.imread(r"C:\Users\Yan\Desktop\a.png", 0)
img2 = cv2.imread(r"C:\Users\Yan\Desktop\b.png", 0)
img3 = cv2.imread(r"C:\Users\Yan\Desktop\c.png", 0)
img4 = cv2.imread(r"C:\Users\Yan\Desktop\128.png", 0)
plt.subplot(131)
plt.imshow(img3, "gray")

plt.subplot(132)
plt.imshow(img1, "gray")

plt.subplot(133)
plt.imshow(img2, "gray")
plt.show()

img4 = cv2.resize(img4, (128, 128), interpolation=cv2.INTER_CUBIC) #双新兴差值法对图像进行缩放
img2 = cv2.resize(img2, (4, 4), interpolation=cv2.INTER_CUBIC) #双新兴差值法对图像进行缩放
plt.imshow(img, "gray")
plt.show()

l = []
mean, stddv = cv2.meanStdDev(img4) #2*2图像， stddv：0.8左右
l.append(stddv)
mean1, stddv1 = cv2.meanStdDev(img1)
l.append(stddv1)
mean2, stddv2 = cv2.meanStdDev(img2)
l.append(stddv2)
print(l)