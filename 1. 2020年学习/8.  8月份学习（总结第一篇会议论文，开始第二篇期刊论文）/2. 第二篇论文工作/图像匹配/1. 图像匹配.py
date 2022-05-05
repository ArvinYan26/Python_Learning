import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('meizhe.jpg', 0)
plt.subplot(211)
plt.imshow(img, "gray")

orb = cv2.ORB_create()


kp = orb.detect(img, None)


kp, des = orb.compute(img, kp)
print(img)
print(len(kp), des)

img = cv2.drawKeypoints(img, kp, img, color=(0, 255, 0), flags=0)

plt.subplot(212)
plt.imshow(img, "gray")
plt.show()