import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain



img = cv2.imread("C:/Users/Yan/Desktop/COVID-19/NORMAL/NORMAL (1).png", cv2.IMREAD_GRAYSCALE)
#img = Image.open("C:/Users/Yan/Desktop/COVID-19/NORMAL/NORMAL (1).png")
print("img:", img.shape)

"""
img = np.array(img)
print("img:", img)
plt.imshow(img)
plt.show()

a = np.array([[1, 2], [3, 4], [9, 8]])
print(type(a))

##使用库函数

a_a = list(chain.from_iterable(a))
print(type(a_a), a_a)
print(type(img))
b = list(chain.from_iterable(img))
print(type(b))
print("img:", img)

"""
x = []
img = np.resize(img, (128, 128))
print("="*10)
print(img.shape[0], img.shape[1])


img = img.reshape(1, -1)
print(type(img), img)
print(type(img[0]), img[0])
print(type(list(img[0])), list(img[0]))
x.append(list(img[0]))
print("x:", x)

a = []
b = []
print(len(img[0]))
for i in range(len(img[0])):
    a.append(img[0][i])
b.append(a)
print(type(b), b)

