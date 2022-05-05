import numpy as np
import networkx as nx
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
from sklearn.model_selection import GridSearchCV
import operator
from normalization import data_preprocess
import BuildNetwork as BN
from DrawGraph import draw_graph
from CaculateMeasures import calculate_measure
from GetSubGraph import GetSubgraph
from Classification import Classification
import time
import pandas as pd

class DataClassification(object):
    """iris数据集分类"""

    def __init__(self, Y_test, k, num_class):
        self.k = k
        # self.g = []
        self.num_class = num_class
        self.Y_test = Y_test
        #self.color_map = {0: 'red', 1: 'green', 2: 'yellow'}
        self.net0_measure = []  # 存储每一类别插入Y_items中的数据时的指标
        self.impact0 = []  # storage the impact of the first class
        self.net1_measure = []
        self.impact1 = []
        self.net2_measure = []
        self.G = None
        result = []

        # 初始化运行程序，一开始就运行
        #self.build_init_network(label=True)
        self.need_classification = []  # 计算模糊分类节点次数
        #self.single_node_insert()
        #self.accuracy()

    def fit(self, X_train, Y_train):
        """获取数据集"""
        self.X_train = X_train
        self.Y_train = Y_train

    def get_params(self, deep=False):
        return {'k': self.k, 'num_class': self.num_class}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X_test, Y_test):
        self.predict(X_test, Y_test)



    def predict(self, X_test, Y_test=[]):
        """

        :param X_items: one node inserted
        :param nodeindex: the index of the inserted node
        :param Y_items: label of the inserted node
        :return:
        """
        #build init network
        self.G = nx.Graph()
        self.G, self.nbrs, radius = BN.build_init_network(self.X_train, self.Y_train,
                                                     self.G, self.k, label=True)
        #draw_graph(self.G)
        GSG = GetSubgraph(self.num_class, self.X_train, self.G)
        self.G0, self.G1, self.G2 = GSG.get_subgraph()
        #steps1：合并小组件
        GSG.merge_components()
        draw_graph(GSG.G)
        #steps2：获得子图
        self.G0, self.G1, self.G2 = GSG.get_subgraph()

        self.X_test = X_test
        result = []
        # 添加节点
        for index, instance in enumerate(self.X_test):
            print("index:", index)
            label = self.Y_test[index]
            print("label:", label)

            if (not len(Y_test) == 0):
                print("label:", self.Y_test[index])
                # print(index, instance)

            # 计算插入节点之前的各个类别网络的measures
            insert_node_id = len(list(self.G.nodes()))
            measures0 = calculate_measure(self.G0)
            self.net0_measure.append(measures0)
            # self.draw_graph(self.G1)
            measures1 = calculate_measure(self.G1)
            self.net1_measure.append(measures1)
            # self.draw_graph(self.G2)
            measures2 = calculate_measure(self.G2)
            self.net2_measure.append(measures2)
            # 插入新的节点构建连边
            BN.node_insert(self.G, self.k, self.nbrs, instance, insert_node_id)
            result = self.classification(insert_node_id, result)
            print("=" * 100)

        #得到最后的图和子图
        draw_graph(self.G)
        print(len(self.G.nodes()))
        GSG = GetSubgraph(self.num_class, self.X_train, self.G)
        self.G0, self.G1, self.G2 = GSG.get_subgraph()
        """
        draw_graph(self.G0)
        draw_graph(self.G1)
        draw_graph(self.G2)
        """
        #得到准确率，和预测target
        label = list(map(int, self.Y_test))  # 廖雪峰，高阶函数内容
        print("original_label:", label)
        print("predict_label :", result)
        count = 0
        for i in range(len(self.Y_test)):
            if self.Y_test[i] == result[i]:
                count += 1
        print("正确个数：", count)
        accuracy = round(count / len(self.Y_test), 3)
        print("accuracy:", accuracy)
        print("need_classification:", self.need_classification)

        return result

    def classification(self, insert_node_id, result):
        # 因为在构建离岸边的时候用的就是str()形式，所以需要此处需要转化为字符串
        adj = [n for n in self.G.neighbors(str(insert_node_id))]  # find the neighbors of the new node
        print("adj:", adj)
        count0 = 0
        count1 = 0
        count2 = 0
        #self.get_subgraph()
        for n in adj:
            if n in self.G0.nodes():
                label = self.G._node[n]["label"]
                #print("label:", label)
                count0 += 1
            elif n in self.G1.nodes():
                label = self.G._node[n]["label"]
                #print("label:", label)
                count1 += 1
            elif n in self.G2.nodes():
                label = self.G._node[n]["label"]
                #print("label:", label)
                count2 += 1
        #print("edges_num:", count0, count1, count2)
        #确认分类后我可以给节点添加标签，以防止新节点与其连接时不知道标签
        if count0 == len(adj):
            print("classification_result:", 0)
            if str(insert_node_id) in self.G.nodes():
                self.G.remove_node(str(insert_node_id))
            self.G.add_node(str(insert_node_id), typeNode='test', label=0)
            for n in adj:
                self.G.add_edge(str(insert_node_id), n)
            result.append(0)
        elif count1 == len(adj):
            print("classification_result:", 1)
            if str(insert_node_id) in self.G.nodes():
                self.G.remove_node(str(insert_node_id))
            self.G.add_node(str(insert_node_id), typeNode='test', label=1)
            for n in adj:
                self.G.add_edge(str(insert_node_id), n)
            result.append(1)
        elif count2 == len(adj):
            print("classification_result:", 2)
            if str(insert_node_id) in self.G.nodes():
                self.G.remove_node(str(insert_node_id))
            self.G.add_node(str(insert_node_id), typeNode='test', label=2)
            for n in adj:
                self.G.add_edge(str(insert_node_id), n)
            result.append(2)
        else:
            print("模糊分类情况：")
            #draw_graph(self.G)
            print(count0, count1, count2)
            dist_list = []
            if count0 == 0:
                dist_list.append(0)
            if count0 > 0 and count0 < len(adj):
                # delate the edges and node
                if str(insert_node_id) in self.G.nodes():
                    self.G.remove_node(str(insert_node_id))
                # 找到类1中和adj中相同的节点，然后将节点添加到类1中
                node_list = self.G0.nodes()  #这时候还是插入节点之前的G0
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类0中
                self.G.add_node(str(insert_node_id), typeNode='test', label=0)
                for n in neighbor:
                    self.G.add_edge(str(insert_node_id), n)
                GSG = GetSubgraph(self.num_class, self.X_train, self.G)
                self.G0, self.G1, self.G2 = GSG.get_subgraph() #get the new sungraph to calclulate the measures
                print(len(self.G0))
                measures0 = calculate_measure(self.G0)  # new subgraph self.G0 measures
                V1, V2 = np.array(self.net0_measure[len(self.net0_measure) - 1]), np.array(measures0)
                print(V1, V2)
                euclidean_dist0 = np.linalg.norm(V2 - V1)
                print(euclidean_dist0)
                dist_list.append(euclidean_dist0)
            if count1 == 0:
                dist_list.append(0)
            if count1 > 0 and count1 < len(adj):
                if str(insert_node_id) in self.G.nodes():
                    self.G.remove_node(str(insert_node_id))
                # 找到类1中和adj中相同的节点，
                node_list = self.G1.nodes()
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类1中
                self.G.add_node(str(insert_node_id), typeNode='test', label=1)
                for n in neighbor:
                    self.G.add_edge(str(insert_node_id), n)
                GSG = GetSubgraph(self.num_class, self.X_train, self.G)

                self.G0, self.G1, self.G2 = GSG.get_subgraph()
                print("len:", len(self.G1))
                measures1 = calculate_measure(self.G1)
                N1, N2 = np.array(self.net1_measure[len(self.net1_measure) - 1]), np.array(measures1)
                print(N1, N2)
                euclidean_dist1 = np.linalg.norm(N2 - N1)
                print(euclidean_dist1)
                dist_list.append(euclidean_dist1)

            if count2 == 0:
                dist_list.append(0)
            if count2 > 0 and count2 < len(adj):
                if str(insert_node_id) in self.G.nodes():
                    self.G.remove_node(str(insert_node_id))
                node_list = self.G2.nodes()
                neighbor = [x for x in node_list if x in adj]
                #添加到2类网络中
                self.G.add_node(str(insert_node_id), typeNode='test', label=2)
                for n in neighbor:
                    self.G.add_edge(str(insert_node_id), n)
                #self.draw_graph(self.G)
                GSG = GetSubgraph(self.num_class, self.X_train, self.G)
                self.G0, self.G1, self.G2 = GSG.get_subgraph()
                print("len:", len(self.G2))
                measures2 = calculate_measure(self.G2)
                M1, M2 = np.array(self.net2_measure[len(self.net2_measure) - 1]), np.array(measures2)
                print("M1, M2:", M1, M2)
                euclidean_dist2 = np.linalg.norm(M2 - M1)
                print(euclidean_dist2)
                dist_list.append(euclidean_dist2)
            #确定模糊分类节点的分类标签后，需要将节点移除，然后重新插入到确定的分类标签处
            while str(insert_node_id) in self.G.nodes(): #防止之前的多次添加，while循环可以删除干净
                self.G.remove_node(str(insert_node_id))
            # print(np.array(self.net0_measure), self.net1_measure, self.net2_measure,)
            print("dist_list:", dist_list)
            # get the classfication ruselt
            list = []
            for x in dist_list:
                if not x == 0:
                    list.append(x)
            min_value = min(list)
            label = int(dist_list.index(min_value))
            print("classification_result:", label)
            result.append(label)

            if label == 0:
                node_list = self.G0.nodes()
                neighbor = [x for x in node_list if x in adj]
                print("neighbor0:", neighbor)
                # 然后将节点添加到类1中
                print("insert_node_id:", insert_node_id)
                for n in neighbor:
                    print(n)
                    self.G.add_node(str(insert_node_id), typeNode='test', label=0)
                    self.G.add_edge(str(insert_node_id), n)  # add edges
                adj = [n for n in self.G.neighbors(str(insert_node_id))]
                print("adj_new:", adj)
            if label == 1:
                node_list = self.G1.nodes()
                neighbor = [x for x in node_list if x in adj]
                print("neighbor1:", neighbor)
                # 然后将节点添加到类1中
                print("insert_node_id:", insert_node_id)
                for n in neighbor:
                    print(n)
                    self.G.add_node(str(insert_node_id), typeNode='test', label=1)
                    self.G.add_edge(str(insert_node_id), n)
                adj = [n for n in self.G.neighbors(str(insert_node_id))]
                print("adj_new:", adj)
            if label == 2:
                node_list = self.G2.nodes()
                neighbor = [x for x in node_list if x in adj]
                print("neighbor2:", neighbor)
                print("insert_node_id:", insert_node_id)

                for n in neighbor:
                    print(n)
                    self.G.add_node(str(insert_node_id), typeNode='test', label=2)
                    self.G.add_edge(str(insert_node_id), str(n))
                adj = [n for n in self.G.neighbors(str(insert_node_id))]
                print("adj:", adj)
            self.need_classification.append(str(insert_node_id))
        return result

if __name__ == '__main__':


    start_time = time.time()
    iris = load_iris()
    X = iris.data  # [:, 2:]
    y = iris.target

    #df = pd.read_csv(r'C:\Users\Yan\Desktop\CovidData_OR.csv')
    """
    features = list(df.columns)
    features = features[: len(features) - 1]  # 去掉开头和结尾的两列数据
    # print(features)
    X = df[features].values.astype(np.float32)
    y = np.array(df.target)
    #print(X, y)
    #print(len(X), len(y))
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    print(len(X_train), len(X_test))
    X_train, X_test = data_preprocess(X_train, X_test)
    DC = DataClassification(Y_test, 4, num_class=3)  # 选择最优的K=2传入模型
    DC.fit(X_train, Y_train)  # 训练模型
    DC.predict(X_test)


