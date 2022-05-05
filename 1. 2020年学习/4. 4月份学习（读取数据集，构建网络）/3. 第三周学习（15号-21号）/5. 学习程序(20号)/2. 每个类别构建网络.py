import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#获取数据集
dataset = load_iris()
print(dataset)
#切分数据
X_train, X_predict, Y_train, Y_predict = train_test_split(dataset["data"], dataset['target'], test_size=0.2)
print(X_train, len(X_train))
print(Y_train, len(Y_train))

#归一化数据
scaler = preprocessing.MinMaxScaler().fit(X_train)
train_data = scaler.transform(X_train)
test_data = scaler.transform(X_predict)
"""
def network_build():
    G = nx.Graph()
    for indices, indececsNode in enumerate(train_data):
"""
