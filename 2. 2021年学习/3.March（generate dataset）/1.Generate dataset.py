import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_gaussian_quantiles

from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

X, y = make_gaussian_quantiles(mean=(0.1, 2), cov=1.0, n_samples=500, n_features=2, n_classes=2, shuffle=True, random_state=None)
plt.style.use('ggplot')
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1],  c=y, alpha=0.7)
plt.subplot(1, 2, 2)
plt.hist(y)
plt.show()


#fig = plt.figure(1, figsize=(20, 10))
color_map = {0: "r", 1: "b"}
for i in [0.05, 0.1, 0.15, 0.2, 0.25]:
    fig = plt.figure(1, figsize=(20, 10))
    x1, y1 = make_circles(n_samples=500, factor=0.02, noise=i)
    print("y1:", y1)
    plt.subplot(121)
    plt.title('make_circles function example')
    plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=y1, linewidths=2)

    plt.subplot(122)
    x1, y1 = make_moons(n_samples=1000, noise=i)
    plt.title('make_moons function example')
    plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=y1)
    plt.show()


