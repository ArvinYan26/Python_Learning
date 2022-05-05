import pyqtree
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\Yan\Desktop\COVID-19\COVID-19\COVID-19 (2).png", 0) #读取灰度图
img1 = cv2.imread(r"C:\Users\Yan\Desktop\resize data\COVID-19\COVID-19 (2).png", 0) #读取灰度图
#print(img1)
plt.subplot(1, 2, 1)
plt.imshow(img, "gray")
plt.subplot(1, 2, 2)
plt.imshow(img1, "gray")
plt.show()

spindex = pyqtree.Index(bbox=[0, 0, 1024, 1024])
spindex.insert(img, )