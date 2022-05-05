import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

iris = pd.read_csv("iris.data", sep=',', header=None, names=None)
iris.columns = ['sepal_len', 'sepal_width', 'petal_len', 'petal_width', 'class']
#header=None:从文本的第一行读取，header=0是从数据的第一行读取(默认为0)，names=None:表示按照原来的文档内容读取出来
iris.head()
print(iris)
iris.describe()

g = nx.Graph()
g.add_nodes_from(iris)
print(g.node)
nx.draw(g)
plt.show()
