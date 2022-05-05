import numpy as np
import normalization as norm
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_iris,load_wine
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import KFold,ShuffleSplit,StratifiedKFold
import pandas as pd
from sklearn import preprocessing

"""
def getDataCSV(url,className):

    dataset = load_iris()
    #keep_default_na=False：源文件中是什么值就显示什么值，不会判定为NaN这些字符。
    #np.nan:生成NaN，na_values=np.nan:将指定的np.nan值视为NaN，也就是见到NaN就判定为缺失值
    data = pd.read_csv(url, keep_default_na=False,  na_values=np.nan)
    if(len(data.values[0])==1):
        data = pd.read_csv(url,";", keep_default_na=False,  na_values=np.nan)
    dataset['target']=data[className].values
    dataset['data']=data.drop(className,axis=1).values
    return dataset
"""
#获取数据集
dataset = load_iris()
print(dataset)

#切分数据集，80%用来训练，20%用来测试
X_train, X_predict, Y_train, Y_predict = train_test_split(dataset['data'], dataset["target"], test_size=0.2)
print(dataset['data'])
##看normalization.py中的内容，导入的是那个文件中的函数preprocess，此处采用的归一化方法是将值归一化成0-1之间的数
normalization=2
link=0
(X_train, X_predict) = norm.preprocess(X_train, X_predict, normalization)

#X_net, X_opt, Y_net, Y_opt= train_test_split(X_train,Y_train,test_size=0.2)

"""
def nodeInsertion(g,nbrs,instance,nodeIndex,label):
    g.add_node(str(nodeIndex), value=instance ,typeNode='test',label=label)
    if(isinstance(instance,(int,float,str))):
        instance=[instance]
    distances,indices = nbrs.kneighbors([instance])  #indice：指数
    for indiceNode,indicesNode in enumerate(indices):
        for tmpi, indice in enumerate(indicesNode):
            g.add_edge(str(indice),str(nodeIndex),weight=distances[indiceNode][tmpi])
    distances,indices = nbrs.radius_neighbors([instance])
    for indiceNode,indicesNode in enumerate(indices):
        for tmpi, indice in enumerate(indicesNode): #indice：指数
            if(not str(indice)==str(indiceNode)):
                g.add_edge(str(indice),str(nodeIndex),weight=distances[indiceNode][tmpi])
"""

def networkBuildKnn(X_net, Y_net, knn, labels=False):  #knn=5,调用传参数为5
    g = nx.Graph()
    lnNet = len(X_net)  #X_net=X_train=0.8*150 = 120，lnNet=120
    g.graph["lnNet"] = lnNet  #训练集网络大小是120
    g.graph["classNames"] = list(set(Y_net))  #训练集类别名，list是将set(Y_net)转化为列表，set是生成一个字典，里面的元素随机排列且不重复
    for index, instance in enumerate(X_net):
        print(index, instance) #带索引的120个训练集的维度值,每个维度一个索引
        g.add_node(str(index), value=instance, typeNode='net', label=Y_net[index])
    values = X_net   #归一化后的值，每一维度的四个特征值都是0-1之间
    #print(instance)  #循环结束只打印最后一次的instance，索引为119
    #print(values)
    #values[0]:数组第一行数据
    if(isinstance(values[0], (int, float, str))):  #判断values[0]是元组中的那一种类型，是的话就返回True
        values = [e[0] for e in values]
    #print(values)

    nbrs= NearestNeighbors(knn+1, metric='euclidean')  #knn=2, 采用欧几里德距离计算knn+1=3个邻居居
    nbrs.fit(values)  #使用values作为训练数据拟合模型
    #print(values)

    distances, indices = nbrs.kneighbors(values) #返回每个点的邻居的距离和索引
    #print(distances)  #距离矩阵中，120各节点，每个节点5个邻居，总共600个邻居，肯定邻居邻居有相同的，只需要把labels相同的连起来
    #print(indices)
    indices = indices[:, 1:]
    #print(indices)
    distances = distances[:, 1:]
    #print(distances)
    #print(distances, indices)
    eRadius = np.quantile(distances, 0.5)  #计算所有距离的分位数，因为是0.5，所以此处是中位数
    #print(eRadius)
    nbrs.set_params(radius=eRadius)
    #print(indices)

    # indceNode:每个节点的索引，从0开始，indicesNode：元素值，即上一步中所有邻居点的索引
    for indiceNode, indicesNode in enumerate(indices):
        #print(indiceNode, indicesNode) #indiceNode:0-119,indicesNode:每个维度5个邻居的矩阵
        for tmpi, indice in enumerate(indicesNode):
            #print(tmpi, indice)
            # tmpi：邻居矩阵的每一个维度的5个邻居的索引，indice：五个邻居中每个邻居的索引（索引的labels就是0,1,2,三个类别）
            if g.nodes()[str(indice)]["label"] == g.nodes()[str(indiceNode)]["label"]:
                g.add_edge(str(indice), str(indiceNode), weight=distances[indiceNode][tmpi])
    print(len(list(g.nodes())))            # 距离是权重，将相同的了labels相同的节点连接起来即可。
    #print(instance)  #循环结束只打印最后一次的instance，索引为119
    distances, indices = nbrs.radius_neighbors([instance]) #查询每个节点，以节点为中心，半径为（radius=eRadius）的单位球体内的邻居
    #print(indices)
    #print(distances)
    for indiceNode, indicesNode in enumerate(indices): #indiceNode:索引， indicesNode:元素
        #print(indiceNode)
        #print(indicesNode)
        for tmpi, indice in enumerate(indicesNode): #tmpi:索引， indice：元素
            #print(tmpi, indice)
            if(not str(indice) == str(indiceNode)):
                if g.nodes()[str(indice)]["label"] == g.nodes()[str(indiceNode)]["label"]:
                    g.add_edge(str(indice), str(indiceNode), weight=distances[indiceNode][tmpi])
    g.graph["index"] = lnNet
    return g, nbrs

if __name__ == "__main__":
    g,nbrs=networkBuildKnn(X_train, Y_train, 5, labels=True)
    nx.draw(g)
    plt.show()
    print(len(list(g.nodes())))
    print(len(list(g.edges())))
                
                

