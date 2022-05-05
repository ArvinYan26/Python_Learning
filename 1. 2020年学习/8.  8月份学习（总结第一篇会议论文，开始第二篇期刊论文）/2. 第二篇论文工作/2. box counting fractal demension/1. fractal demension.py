import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
import cv2 as cv


def rgb2gray(rgb):  #转化为灰度图像，3通道
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def fractal_dimension(Z, threshold):
    # Only for 2d image
    assert(len(Z.shape) == 2) #assert ： 判断语句，如果是二维图像就执行，不是的话，直接报错，不再执行

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        #axis=0, 按列计算， 1：按行计算
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    # Transform Z into a binary array
    Z = (Z < threshold)
    #plt.draw()
    #plt.show()

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

I = cv.imread("C:/Users/Yan/Desktop/COVID-19/NORMAL/NORMAL (3).png", cv.IMREAD_GRAYSCALE)
I1 = cv.imread("C:/Users/Yan/Desktop/COVID-19/NORMAL/NORMAL (8).png", cv.IMREAD_GRAYSCALE)
I2 = cv.imread("C:/Users/Yan/Desktop/COVID-19/COVID-19/COVID-19 (4).png", cv.IMREAD_GRAYSCALE)
I3 = cv.imread("C:/Users/Yan/Desktop/COVID-19/COVID-19/COVID-19 (5).png", cv.IMREAD_GRAYSCALE)
I4 = cv.imread("C:/Users/Yan/Desktop/COVID-19/Viral Pneumonia/Viral Pneumonia (5).png", cv.IMREAD_GRAYSCALE)
I5 = cv.imread("C:/Users/Yan/Desktop/COVID-19/Viral Pneumonia/Viral Pneumonia (6).png", cv.IMREAD_GRAYSCALE)

"""
I = cv.imread("C:/Users/Yan/Desktop/COVID-19/NORMAL/NORMAL (15).png", cv.IMREAD_GRAYSCALE)
I1 = cv.imread("C:/Users/Yan/Desktop/COVID-19/NORMAL/NORMAL (18).png", cv.IMREAD_GRAYSCALE)
I2 = cv.imread("C:/Users/Yan/Desktop/COVID-19/COVID-19/COVID-19 (9).png", cv.IMREAD_GRAYSCALE)
I3 = cv.imread("C:/Users/Yan/Desktop/COVID-19/COVID-19/COVID-19 (26).png", cv.IMREAD_GRAYSCALE)
I4 = cv.imread("C:/Users/Yan/Desktop/COVID-19/Viral Pneumonia/Viral Pneumonia (40).png", cv.IMREAD_GRAYSCALE)
I5 = cv.imread("C:/Users/Yan/Desktop/COVID-19/Viral Pneumonia/Viral Pneumonia (500).png", cv.IMREAD_GRAYSCALE)
#print(I2[500][250])
"""

Threshold = [x for x in range(100, 160, 2)]
print(len(Threshold))
D = []
D1 = []
D2 = []
D3 = []
D4 = []
D5 = []

for x in Threshold:
    #print("Minkowski–Bouligand dimension (computed): ", fractal_dimension(I, i))
    y = fractal_dimension(I, x)
    y1 = fractal_dimension(I1, x)
    y2 = fractal_dimension(I2, x)
    y3 = fractal_dimension(I3, x)
    y4 = fractal_dimension(I4, x)
    y5 = fractal_dimension(I5, x)
    D.append(y)
    D1.append(y1)
    D2.append(y2)
    D3.append(y3)
    D4.append(y4)
    D5.append(y5)


print("Threshold:", len(Threshold), Threshold)
print("D:", len(D), D)
print("D1:", len(D1), D1)
print("D2:", len(D2), D2)
print("D3:", len(D3), D3)


plt.figure("features")
plt.plot(Threshold, D, color="red", label="NORMAL3")
plt.plot(Threshold, D1, color="red", label="NORMAL8")
print(len(Threshold), len(D2))
plt.plot(Threshold, D2, color="purple", label="COVID-194")
plt.plot(Threshold, D3, color="purple", label="COVID-195")
plt.plot(Threshold, D4, color="green", label="Viral Pneumonia5")
plt.plot(Threshold, D5, color="green", label="Viral Pneumonia6")
plt.legend()
plt.xlabel("Thershold")
plt.ylabel("Desmension")
#plt.draw()
plt.show()
#plt.legend()
#plt.pause(5) #显示5秒
#plt.savefig("box, counting fractal demension")
#plt.close() #关闭