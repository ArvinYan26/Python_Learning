import numpy as np
from sklearn.datasets import load_iris
"""
#随机取出来一个多维数据0.2比例的几行数据作为测试集
iris= load_iris()
iris_data = iris.data
iris_target = iris.target
#print(iris_data, iris_target)
print(len(iris_data))

#测试集比例
class_num = 3
train_size = 0.2
data_len = len(iris_data)
per_class_len = data_len/class_num  #150/3 = 50
target_len = len(iris_target)
"""

def split_data(data, data_target, X_train, Y_train, train_data, train_target):
    """
    将元数据随机划分成训练集和测试集
    :param data:  需要划分的总的数据集
    :param data_target: 需要划分的总的数据集对应的label
    :param data_len: 需要划分的数据长度
    :param per_class_len: 需要划分的数据集中每一类的长度（数量）
    :param X_train: 用来存储训练集的空列表
    :param Y_train:  用来存储训练集相应数据的label的空列表
    :return:
    """
    class_num = 3
    train_size = 0.2 #测试集大小
    data_len = len(data)  #总的数据及长度
    per_class_len = data_len / class_num  # 150/3 = 50 ， 每一类数据集长度

    li0 = [] #用来存储随机抽出来的测试集数据索引
    for j in range(class_num):  #循环每一类数据，然后取测试集数据
        #li0 = []  #抽取的数据行索引
        i = 0
        while i < train_size*data_len/class_num: #0.2*150/3 每一类抽出来的数据个数
            row = np.random.randint(per_class_len*j, per_class_len*(j+1))  #在每一类中产生随机索引值，0-50， 50-100， 100-150
            if row not in li0:
                li0.append(row)
                i += 1
        #print("li0:", li0)
        #print("li0_len:", len(data))
        #print("li0_len:", len(li0))
    for i in li0:
        X_train.append(data[i])   #测试集大小
        Y_train.append(data_target[i])  #测试集label


    #这里的li0表示已经划分出来的测试集数据索引，然后传给delete函数，在原数据及上删除这些指定的索引元素，剩下来的就是训练集
    #delete函数可以指定删除一行，也可以指定删除多行元素，但是不能放在for循环里面，一次删除一个，剩下的元素的索引已经变了，不再是原来的索引
    #再继续想删除原来的元素的时候就会出错

    #train_data:训集
    train_data = np.delete(data, li0, axis=0)   #i:如果i=1那么久表示删除的市第二行元素， i直接表示行号索引； 这里li0表示直接同时删除多行指定的元素
    #train_target:训练集标签
    train_target = np.delete(data_target, li0, axis=0)
    """
    print("X_train:", np.array(X_train))
    print("X_train_len", len(X_train))
    print("Y_train", np.array(Y_train))
    print("Y_train_len", len(Y_train))

    print(train_data, train_target)
    print(len(train_data))
    print(len(train_target))
    """

    return train_data, train_target, X_train, Y_train

"""
X_train1 = []
Y_train1 = []
X_train2 = []
Y_train2 = []
train_data = []
train_target = []

X_train1, Y_train1 = split_data(iris_data, iris_target, data_len, per_class_len, X_train1, Y_train1)
print("训练据集：")
print(np.array(X_train1), np.array(Y_train1))

date_len = len(X_train1)
pre_class_len = len(X_train1)/3
X_train2, Y_train2 = split_data(X_train1, Y_train1, date_len, pre_class_len, X_train2, Y_train2)
print("测试集：")
print(np.array(X_train2), np.array(Y_train2))
"""