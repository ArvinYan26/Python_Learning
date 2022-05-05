from sklearn.decomposition import PCA
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def draw_graph(img):
    plt.imshow(img)
    plt.show()



im = cv.imread("C:/Users/Yan/Desktop/COVID-19/NORMAL/NORMAL (1).png")###导入图片
plt.title("original_img")
draw_graph(im)

imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)###将彩色图片灰度化
draw_graph(imgray)

pca = PCA(n_components=0.95)  ###PCA，提取特征为256
p = pca.fit_transform(imgray) ###对原始数据进行降维

print("im:", im.shape)
print("imgray:", imgray.shape)
print("p:", p)
print("p.shape:", p.shape)


#一张图片一个向量
imag1 = cv.imread("C:/Users/Yan/Desktop/COVID-19/NORMAL/NORMAL (1).png", cv.COLOR_BGR2GRAY)
imag2 = cv.imread("C:/Users/Yan/Desktop/COVID-19/NORMAL/NORMAL (2).png", cv.COLOR_BGR2GRAY)
imag3 = cv.imread("C:/Users/Yan/Desktop/COVID-19/Viral Pneumonia/Viral Pneumonia (1).png", cv.COLOR_BGR2GRAY)


#print(type(imag1))
draw_graph(imag1)
imag1 = imag1.reshape(1, -1)
print("imag:", len(imag1[0]), imag1)

imag2 = imag2.reshape(1, -1)
print("imag3:", imag3)
imag3 = imag3.reshape(1, -1)

all_imag = np.append(imag1, imag2, axis=0)
all_imag = np.append(all_imag, imag3, axis=0)
print(all_imag)

pc = PCA(n_components=0.95)
all_imag = pc.fit_transform(all_imag)
print("all_imag:", all_imag)


