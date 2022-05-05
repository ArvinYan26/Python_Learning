import imageio
import numpy as np
import matplotlib.pylab as plt

def median_polling(img_path):
    img = imageio.imread(img_path)
    s = 8
    rows, cols, C = img.shape

    nrows = int(rows/s)
    ncols = int(cols/s)
    img1 = np.zeros((nrows, ncols, C), np.uint8)

    for row in range(nrows):
        for col in range(ncols):
            for c in range(C):
                img1[row,  col, c] = np.median(img[row*s:(row+1)*s-1, col*s:(col+1)*s-1, c])

    return img1


