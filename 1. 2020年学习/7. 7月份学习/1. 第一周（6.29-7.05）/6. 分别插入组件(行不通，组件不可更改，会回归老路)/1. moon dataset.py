from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from main import DataClassification
from normalization import data_preprocess
from sklearn.model_selection import train_test_split
from itertools import cycle, islice
import numpy as np
from sklearn.datasets import make_moons

n_samples = 200
def make_moon():
    X, Y = make_moons(n_samples=n_samples, noise=0.4)
    #print(X, Y)
    x = X[:, 0]
    y = X[:, 1]
    plt.title("Moon(noise=0.25)")
    plt.scatter(x, y, s=10, c=Y)
    plt.show()
    return X, Y

def make_circle():
    X1, Y1 = make_circles(n_samples=n_samples, noise=0.25)
    #print(X1, Y1)
    x = X1[:, 0]
    y = X1[:, 1]
    plt.title("Circle(noise=0.25)")
    plt.scatter(x, y, s=10, c=Y1)
    plt.show()
    return X1, Y1
def make_blob():
    X2, Y2 = make_blobs(n_samples=n_samples)
    #print(X2, Y2)
    x = X2[:, 0]
    y = X2[:, 1]
    plt.title("Blobs(noise=0.0)")
    plt.scatter(x, y, s=10, c=Y2)
    plt.show()
    #train_data = scaler.transform(X)
    #test_data = scaler.transform(data_test)
    return X2, Y2

def main(X, Y, k, num_class):
    DC = DataClassification(k=k, num_class=num_class)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_test = data_preprocess(X_train, X_test, 2)
    DC.fit(X_train, Y_train)
    acc = DC.predict(X_test, Y_test)
    return acc

if __name__ == '__main__':
    l = []
    X, Y = make_moon()
    X1, Y1 = make_circle()
    X2, Y2 = make_blob()
    acc = main(X, Y, 2, 2)
    l.append(acc)
    acc1 = main(X1, Y1, 2, 2)
    l.append(acc1)
    #acc2 = main(X2, Y2)
    #l.append(2)
    print(l)
