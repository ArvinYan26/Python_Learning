import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage.measure import label

def plt_img(img):
    plt.imshow(img, "gray")

    #不显示坐标
    #plt.xticks([])
    #plt.yticks([])
    plt.show()

#img = cv2.imread(r"C:\Users\Yan\Desktop\testdata\COVID-19\COVID-19 (1).png", 0)
"""
#print("img:", img)
plt_img(img)

g_img = cv2.GaussianBlur(img, (5, 5), 0)
plt_img(g_img)

ret, b_img = cv2.threshold(g_img, 100, 255, cv2.THRESH_BINARY)
plt_img(b_img)

#清除边界
cl_img = clear_border(b_img)
plt_img(cl_img)

"""
im = cv2.imread(r"C:\Users\Yan\Desktop\testdata\COVID-19\COVID-19 (1).png", 0)
blur = cv2.GaussianBlur(im, (5, 5), 0)
ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.figure(num=8, figsize=(20, 20))

'''
Step 1: 二值化
'''
binary = im < ret  # 还可以直接打印矩阵th,对应的就是二值化后的图像，注意这里面使用的binary是布尔逻辑值。
ax = plt.subplot(331)
ax.axis('off')
ax.set_title('Step 1')
ax.imshow(binary, cmap=plt.cm.bone)

'''
Step 2: 清除边界（from skimage.segmentation import clear_border）
'''
cleared = clear_border(binary)
ax = plt.subplot(332)
ax.axis('off')
ax.set_title('Step 2')
ax.imshow(cleared, cmap=plt.cm.bone)

'''
Step 3: 膨胀操作.
'''
selem = disk(2)
cleared = dilation(cleared, selem)
ax = plt.subplot(333)
ax.axis('off')
ax.set_title('Step 3')
ax.imshow(cleared, cmap=plt.cm.bone)

'''
Step 4: 连通区域标记（from skimage.measure import label很好使用）.
'''
label_image = label(cleared)
ax = plt.subplot(334)
ax.axis('off')
ax.set_title('Step 4')
ax.imshow(label_image, cmap=plt.cm.bone)

'''
Step 5: 寻找最大的两个连通区域：肺区.
'''
areas = [r.area for r in regionprops(label_image)]
areas.sort()
if len(areas) > 2:
    for region in regionprops(label_image):
        if region.area < areas[-2]:
            for coordinates in region.coords:
                label_image[coordinates[0], coordinates[1]] = 0
binary = label_image > 0
ax = plt.subplot(335)
ax.axis('off')
ax.set_title('Step 5')
ax.imshow(binary, cmap=plt.cm.bone)

'''
Step 6: 腐蚀操作.
'''
selem = disk(2)
binary = binary_erosion(binary, selem)
ax = plt.subplot(336)
ax.axis('off')
ax.set_title('Step 6')
ax.imshow(binary, cmap=plt.cm.bone)

'''
Step 7: 闭合操作.
'''
selem = disk(10)
binary = binary_closing(binary, selem)
ax = plt.subplot(337)
ax.axis('off')
ax.set_title('Step 7')
ax.imshow(binary, cmap=plt.cm.bone)

'''
Step 8: 孔洞填充.
'''
edges = roberts(binary)
binary = ndi.binary_fill_holes(edges)
ax = plt.subplot(338)
ax.axis('off')
ax.set_title('Step 8')
ax.imshow(binary, cmap=plt.cm.bone)

'''
Step 9: 生成ROI区域.
'''
get_high_vals = binary == 0
im[get_high_vals] = 0
ax = plt.subplot(339)
ax.axis('off')
ax.set_title('Step 9')
ax.imshow(im, cmap=plt.cm.bone)
plt.show()
