from sklearn.datasets import load_iris, load_wine
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from DrawGraph import draw_graph
from CaculateMeasures import calculate_measure
from ReorganizeData import reorganize_data
from sklearn.model_selection import train_test_split
from normalization import data_preprocess
from BuildNetwork import BuildNetwork
from GetSubgraph import GetSubgraph
from CaculateMeasures import calculate_measure
from sklearn import preprocessing

class Classification(object):

    def __init__(self, x, class_num):
        self.class_num = class_num
        self.x = x
        #self.k = k
        self.measures0 = []
        self.measures1 = []
        self.measures2 = []
        self.mean_l = [] #存储初始化每类网络的内的相似度均值

    def fit(self, X_train, Y_train):
        """获取数据集"""
        self.X_train = X_train
        self.Y_train = Y_train

    def get_params(self, deep=False):
        return {'k': self.k, 'num_class': self.class_num}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X_test, Y_test):
        self.predict(X_test, Y_test)

    def calculate_dis(self, G, insert_node_id, measures):
        """
        data = [self.x[i] for i in G.nodes()]
        data = np.vstack((data, self.x[insert_node_id]))
        t = []
        for i in range(len(data)):
            for j in range(i+1, len(data)):
                dis = np.linalg.norm(data[i] - data[j])
                t.append(dis)
        mean1 = np.mean(t)
        print("mean1:", mean1, t)
        #print(len(data), len(self.x), insert_node_id)
        #count = 0
        t2 = []
        for i in G.nodes():
            #print("i:", i)
            #count += 1
            #print("count:", count)
            if not i == insert_node_id:
                dis = np.linalg.norm(np.array(self.x[i]) - self.x[insert_node_id])  # x[insert_node_id]:新插入的数据在原始数据集中的位置数据
                t2.append(dis)
        print("t2:", len(t2))
        mean_val = np.mean(t2)
        """
        s = []
        t1 = []
        for i in G.nodes():
            #print("i:", i)
            #count += 1
            #print("count:", count)
            if not i == insert_node_id:
                dis = np.linalg.norm(np.array(self.x[i]) - self.x[insert_node_id])  # x[insert_node_id]:新插入的数据在原始数据集中的位置数据
                t1.append(dis)
                #print(dis, measures[len(measures)-1])
                #if dis < mean_val:
                if dis < measures[len(measures)-1]:
                    self.G.add_edge(i, insert_node_id, weight=dis)
                    s.append(i) #存储新节点邻居的节点编号
        # 计算新来的数据和所有过去数据的平均相似度（着重强调新插入的数据）
        #print("="*10)
        mean2 = np.mean(t1)
        print("mean2:", mean2, t1)
        print(s)
        return s


    def predict(self, X_test, Y_test):
        #print(self.X_train)
        self.X_test = X_test
        self.X_train, self.Y_train, self.mean_l, data_len = reorganize_data(self.X_train, self.Y_train)
        #最初的那个同类之间的相似度均值可能后边用到，所以设为全局
        print(self.mean_l, data_len, len(self.X_train))
        #print(self.X_train, self.Y_train)

        B = BuildNetwork()
        self.G = B.build_init_network(self.X_train, self.Y_train, self.mean_l, data_len, self.class_num)
        draw_graph(self.G)

        #初始化网络的measures
        GS = GetSubgraph(self.G, self.class_num)
        self.G0, self.G1, self.G2 = GS.get_subgraph() #得到子图
        measures0 = calculate_measure(self.G0, self.X_train, [])
        self.measures0.append(measures0)
        measures1 = calculate_measure(self.G1, self.X_train, [])
        self.measures1.append(measures1)
        measures2 = calculate_measure(self.G2, self.X_train, [])
        self.measures2.append(measures2)
        print("+"*10)
        print(self.measures0, self.measures1, self.measures2)



        #分类阶段
        for idx, instance in enumerate(self.X_test):
            print("label:", Y_test[idx])
            GS = GetSubgraph(self.G, self.class_num)
            self.G0, self.G1, self.G2 = GS.get_subgraph()  # 得到子图
            print(len(self.G0.nodes), len(self.G1.nodes), len(self.G2.nodes))

            insert_node_id = len(self.G.nodes())
            #添加新节点
            self.G.add_node(insert_node_id, value=instance, typeNode="test")
            print(len(self.G.nodes))
            dis_cam = []
            #插入到网络1中
            self.adj0 = self.calculate_dis(self.G0, insert_node_id, self.measures0[len(self.measures0)-1])
            print(len(self.G.nodes), self.adj0)
            GS = GetSubgraph(self.G, self.class_num)
            self.G0, self.G1, self.G2 = GS.get_subgraph()  # 得到子图
            print(len(self.G0.nodes), len(self.G.nodes))
            measures0 = calculate_measure(self.G0, self.x, insert_node_id)
            print("measures0:", measures0)
            dis0 = np.linalg.norm(np.array(self.measures0[len(self.measures0)-1]) - measures0)
            dis_cam.append(dis0)
            #draw_graph(self.G)

            #插入网络2中
            if insert_node_id in self.G.nodes():
                self.G.remove_node(insert_node_id)
            self.G.add_node(insert_node_id, value=instance, typeNode="test")
            print("G1:", len(self.G1.nodes))
            self.adj1 = self.calculate_dis(self.G1, insert_node_id, self.measures1[len(self.measures1)-1])
            print(self.adj1)
            GS = GetSubgraph(self.G, self.class_num)
            self.G0, self.G1, self.G2 = GS.get_subgraph()  # 得到子图
            print(len(self.G1.nodes))
            measures1 = calculate_measure(self.G1, self.x, insert_node_id)
            print("measures1:", measures1)
            dis1 = np.linalg.norm(np.array(self.measures1[len(self.measures1)-1]) - measures1)
            dis_cam.append(dis1)

            #插入网络3中
            if insert_node_id in self.G.nodes():
                self.G.remove_node(insert_node_id)
            self.G.add_node(insert_node_id, value=instance, typeNode="test")
            self.adj2 = self.calculate_dis(self.G2, insert_node_id, self.measures2[len(self.measures2)-1])
            print(self.adj2)
            GS = GetSubgraph(self.G, self.class_num)
            self.G0, self.G1, self.G2 = GS.get_subgraph()  # 得到子图
            print(len(self.G2.nodes))
            measures2 = calculate_measure(self.G2, self.x, insert_node_id)
            print("measures2:", measures2)
            dis2 = np.linalg.norm(np.array(self.measures2[len(self.measures2)-1]) - measures2)
            dis_cam.append(dis2)
            #draw_graph(self.G)
            print("dis_cam:", dis_cam), print(Y_test[idx])

            #最后移除该节点
            self.G.remove_node(insert_node_id)
            break

if __name__ == '__main__':

    #data = load_wine()
    #data = load_iris()
    #print(data)
    #x = data.data
    #y = data.target

    df = pd.read_csv(r"C:\Users\Yan\Desktop\CovidData_n12.csv")

    features = list(df.columns)
    features = features[: len(features) - 1]  # 去掉开头和结尾的两列数据
    # print(features)
    x = df[features].values.astype(np.float32)
    y = np.array(df.target)

    x = data_preprocess(x) #直接将所有数据预处理，然后进行切分
    #print(x)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1)
    c = Classification(x, class_num=3)
    c.fit(X_train, Y_train)
    c.predict(X_test, Y_test)

