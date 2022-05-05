import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from  sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors


#随机取出来一个多维数据0.2比例的几行数据作为测试集



iris= load_iris()
iris_data = iris.data
iris_target = iris.target
#print(iris_data, iris_target)
print(len(iris_data))

#测试集比例
class_num = 3
train_size = 0.2
X_net_size = 0.8    #items:0.2
data_len = len(iris_data)
per_class_len = data_len/class_num  #150/3 = 50
target_len = len(iris_target)

X_train1 = []
Y_train1 = []
X_train2 = []
Y_train2 = []
X_train3 = []
Y_train3 = []


def split_data(data, data_target, data_len, per_class_len, X_train, Y_train):
    print(data_len, per_class_len, train_size, class_num)
    #for i in range(class_num):
    for j in range(class_num):
        li0 = []  #抽取的数据行索引
        i = 0
        while i < train_size*data_len/class_num: #0.2*150/3 每一类抽出来的数据个数
            row = np.random.randint(per_class_len*j, per_class_len*(j+1))
            #print(type(row))
            if row not in li0:
                li0.append(row)
                i += 1
        print("li0:", li0)
        #print("li0_len:", len(data))
        print("li0_len:", len(li0))
        for i in li0:
            X_train.append(data[i])
            Y_train.append(data_target[i])
            #data = np.delete(data, i, axis=0)   #i:如果i=1那么久表示删除的市第二行元素， i直接表示行号索引
            #data_target = np.delete(data_target, iris_target[i], axis=0)

        print("X_train1:", np.array(X_train1))
        print("X_train1_len", len(X_train1))
        print("Y_train1", np.array(Y_train1))
        print("Y_train1_len", len(Y_train1))

        #print("len:", len(iris_data))
        #print("iris_target:", len(iris_target))

    return X_train, Y_train
X_train1, Y_train1 = split_data(iris_data, iris_target, data_len, per_class_len, X_train1, Y_train1)
print("训练据集：")
print(np.array(X_train1), np.array(Y_train1))


X_train2, Y_train2 = split_data(X_train1, Y_train1, len(X_train1), len(X_train1)/3, X_train2, Y_train2)
print("新的训练集：")
print(np.array(X_train2), np.array(Y_train2))
