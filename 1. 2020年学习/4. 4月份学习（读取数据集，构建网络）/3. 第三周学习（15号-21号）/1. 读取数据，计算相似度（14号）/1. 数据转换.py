import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


iris = pd.read_csv("iris.csv", sep=',', header=0, names=None)
print(iris)
#iris_data = iris[1:, 2:5]
#print(iris_data)
iris.as_matrix(columns=None)
#iris.columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
#x = iris[["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]]
#x = np.array(x)
print(iris)

#header=None:从文本的第一行读取，header=0是从数据的第一行读取(默认为0)，names=None:表示按照原来的文档内容读取出来
iris.head()
print(iris)

"""
iris.data = pd.read_csv("iris.data", sep=',')
iris.head()
print(iris)

df = pd.read_csv("iris_data1.txt", sep='\t')

print(iris1)

data = np.loadtxt('iris_data1.txt')
print(data)
"""