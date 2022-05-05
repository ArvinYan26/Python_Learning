from sklearn.decomposition import PCA
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def draw_graph(img):
    plt.imshow(img)
    plt.show()

def reshape(img):
    img = cv.resize(img, (54, 54))
    draw_graph(img)
    img = img.reshape(1, -1)
    return img

imag1 = cv.imread("C:/Users/Yan/Desktop/data/NORMAL/NORMAL (1).png", cv.COLOR_BGR2GRAY)
imag2 = cv.imread("C:/Users/Yan/Desktop/data/NORMAL/NORMAL (2).png", cv.COLOR_BGR2GRAY)
imag3 = cv.imread("C:/Users/Yan/Desktop/data/Viral Pneumonia/Viral Pneumonia (1).png", cv.COLOR_BGR2GRAY)
imag4 = cv.imread("C:/Users/Yan/Desktop/data/Viral Pneumonia/Viral Pneumonia (2).png", cv.COLOR_BGR2GRAY)

"""
pca = PCA(n_components=20)
imag1 = pca.fit_transform(imag1)
imag2 = pca.fit_transform(imag2)
imag3 = pca.fit_transform(imag3)


print(imag1.shape, imag1)
print(imag2.shape, imag2)
print(imag3.shape, imag3)
"""

imag1 = reshape(imag1)
imag2 = reshape(imag2)
imag3 = reshape(imag3)
imag4 = reshape(imag4)

print(imag1.shape, imag1)
print(imag2.shape, imag2)
print(imag3.shape, imag3)

all_dis = []
dist1 = np.linalg.norm(imag1-imag2)
print(dist1)
all_dis.append(dist1)

dist2 = np.linalg.norm(imag1-imag3)
print(dist2)
all_dis.append(dist2)

dist3 = np.linalg.norm(imag1-imag4)
print(dist3)
all_dis.append(dist3)

dist4 = np.linalg.norm(imag2-imag3)
print(dist4)
all_dis.append(dist4)

dist5 = np.linalg.norm(imag2-imag4)
print(dist5)
all_dis.append(dist5)

dist6 = np.linalg.norm(imag3-imag4)
print(dist6)
all_dis.append(dist3)

print("all_dis:", all_dis)

"""
l = []
l.append(imag1[0])
l.append(imag2[0])
l.append(imag3[0])
l.append(imag4[0])
"""
l = []
imag1 = list(imag1[0])
imag2 = list(imag2[0])
imag3 = list(imag3[0])
imag4 = list(imag4[0])
l.append(imag1)
l.append(imag2)
l.append(imag3)
l.append(imag4)
print(np.array(l))
