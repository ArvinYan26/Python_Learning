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


def getDataCSV(url,className):
    dataset = load_iris()
    #keep_default_na=False：源文件中是什么值就显示什么值，不会判定为NaN这些字符。

    data = pd.read_csv(url, keep_default_na=False,  na_values=np.nan)
    print(data)
    if(len(data.values[0])==1):
        data = pd.read_csv(url,";", keep_default_na=False,  na_values=np.nan)
    dataset['target']=data[className].values
    dataset['data']=data.drop(className,axis=1).values
    return dataset
#%%
dataset = load_iris()
X_train, X_predict, Y_train, Y_predict= train_test_split(dataset['data'],dataset["target"],test_size=0.2)
knn=2
normalization=2
link=0
(X_train,X_predict)=norm.preprocess(X_train,X_predict,normalization)

#X_net, X_opt, Y_net, Y_opt= train_test_split(X_train,Y_train,test_size=0.2)

#%%
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
def networkBuildKnn(X_net,Y_net,knn,labels=False):
    g=nx.Graph()
    lnNet=len(X_net)
    g.graph["lnNet"]=lnNet
    g.graph["classNames"]=list(set(Y_net))
    for index,instance in enumerate(X_net):
        g.add_node(str(index), value=instance ,typeNode='net',label=Y_net[index])
    values=X_net
    
    if(isinstance(values[0],(int,float,str))):
        values=[e[0] for e in values]
        
    nbrs= NearestNeighbors(knn+1,metric='euclidean')
    nbrs.fit(values)

    distances,indices = nbrs.kneighbors(values)
    indices=indices[:, 1:]
    distances=distances[:, 1:]
    eRadius=np.quantile(distances,0.5)
    nbrs.set_params(radius=eRadius)
    
    for indiceNode,indicesNode in enumerate(indices):
        for tmpi, indice in enumerate(indicesNode):
            if( g.nodes()[str(indice)]["label"] == g.nodes()[str(indiceNode)]["label"] or not labels):
                g.add_edge(str(indice),str(indiceNode),weight=distances[indiceNode][tmpi])
    
    distances,indices = nbrs.radius_neighbors([instance])
    for indiceNode,indicesNode in enumerate(indices):
        for tmpi, indice in enumerate(indicesNode):
            if(not str(indice)==str(indiceNode)):
                if( g.nodes()[str(indice)]["label"] == g.nodes()[str(indiceNode)]["label"] or not labels):
                    g.add_edge(str(indice),str(indiceNode),weight=distances[indiceNode][tmpi])
    g.graph["index"]=lnNet
    return g,nbrs
#%%
g,nbrs=networkBuildKnn(X_train, Y_train, 5,labels=True)
nx.draw(g)
plt.show()
print(len(list(g.nodes())))
print(len(list(g.edges())))

#getDataCSV()
                

