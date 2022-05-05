import numpy as np
import networkx as nx
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
from sklearn.model_selection import GridSearchCV
import operator
from normalization import data_preprocess
import BuildNetwork as BN
from DrawGraph import draw_graph
from CaculateMeasures import calculate_measure
import pandas as pd
import time


class DataClassification(object):
    """iris数据集分类"""

    def __init__(self, k, num_class):
        self.k = k
        # self.g = []
        self.num_class = num_class
        self.color_map = {1: 'red', 2: 'green', 3: 'yellow'}
        self.net0_measure = []  # 存储每一类别插入Y_items中的数据时的指标
        self.impact0 = []  # storage the impact of the first class
        self.net1_measure = []
        self.impact1 = []
        self.net2_measure = []
        self.G = None
        self.predict_label = []

        # 初始化运行程序，一开始就运行
        #self.build_init_network(label=True)
        self.need_classification = []  # 计算模糊分类节点次数
        #self.single_node_insert()
        #self.accuracy()

    def fit(self, X_train, X_test, Y_train, Y_test):
        """获取数据集"""
        self.X_train = X_train
        self.Y_train = Y_train

    def get_subgraph(self):
        """得到子图，并画出来"""
        num = nx.number_connected_components(self.G)
        #print("num_components:", num)
        Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)
        #list = []
        for n in range(num):
            G = self.G.subgraph(Gcc[n])
            count0 = 0
            count1 = 0
            count2 = 0
            for m in G.nodes():
                if G._node[m]["label"] == 1:
                    count0 += 1
                    if count0 > 3:  #设置阈值，查找大的组件构建初试类网络，用以后边吞并小的组件
                        self.G0 = G
                if G._node[m]["label"] == 2:
                    count1 += 1
                    if count1 > 3:
                        self.G1 = G
                if G._node[m]["label"] == 3:
                    count2 += 1
                    if count2 > 3:
                        self.G2 = G

        # 如果组件数大于类别数执行下面步骤,正常情况下，分类阶段不会用到，因为很明显的分3类
        if num > self.num_class:
            for m in range(num):
                G = self.G.subgraph(Gcc[m])
                for n in G.nodes():
                    if G._node[n]["label"] == 1:
                        count = 0
                        for a in self.G0.nodes():
                            if n == a:
                                continue
                            self.G.add_edge(str(n), str(a))
                            count += 1
                            if count == self.k:  #
                                break
                    if G._node[n]["label"] == 2:
                        count = 0
                        for a in self.G1.nodes():
                            if n == a:
                                continue
                            self.G.add_edge(str(n), str(a))
                            count += 1
                            if count == self.k:
                                break
                    if G._node[n]["label"] == 3:
                        count = 0
                        for a in self.G2.nodes():
                            if n == a:
                                continue
                            self.G.add_edge(str(n), str(a))
                            count += 1
                            if count == self.k:
                                break
    def get_params(self, deep=False):
        return {'k': self.k, 'num_class': self.num_class}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, X_test, Y_test):
        """

        :param X_items: one node inserted
        :param nodeindex: the index of the inserted node
        :param Y_items: label of the inserted node
        :return:
        """
        # print(self.X_test, self.Y_test)
        # g = self.g
        # print(len(g.nodes()))
        # insert_node_id = len(list(self.g.nodes()))
        # print(insert_node_id)
        self.G = nx.Graph()
        self.G, nbrs, radius = BN.build_init_network(self.X_train, self.Y_train,
                                                               self.G, self.k, label=True)

        # print("nodes:", self.G.nodes())
        self.get_subgraph()
        #draw_graph(self.G)
        self.get_subgraph()
        draw_graph(self.G)
        self.X_test = X_test
        self.Y_test = Y_test
        print("length_X_trian:", len(self.X_train))
        print("length_X_test:", len(self.Y_test))
        # 添加节点
        for index, instance in enumerate(self.X_test):
            # label = 4
            label = self.Y_test[index]
            #print("label:", label)
            if (not len(self.Y_test) == 0):
                label = self.Y_test[index]
                # print(index, instance)
            # 计算插入节点之前的各个类别网络的measures
            self.get_subgraph()
            # self.draw_graph(self.G0)
            measures0 = calculate_measure(self.G0)
            self.net0_measure.append(measures0)
            # self.draw_graph(self.G1)
            measures1 = calculate_measure(self.G1)
            self.net1_measure.append(measures1)
            # self.draw_graph(self.G2)
            measures2 = calculate_measure(self.G2)
            self.net2_measure.append(measures2)
            # 插入新的节点构建连边
            insert_node_id = len(list(self.G.nodes()))
            #print("insert_node_id:", insert_node_id)
            BN.node_insert(self.G, self.k, nbrs, instance, insert_node_id, label)
            self.classification(insert_node_id, int(self.Y_test[index]))
            #print("=" * 100)

        #得到最后的图和子图
        draw_graph(self.G)
        #self.get_subgraph()


        #得到准确率，和预测target
        label = list(map(int, self.Y_test))  # 廖雪峰，高阶函数内容
        print("original_label:", label)
        print("predict_label :", self.predict_label)

        count = 0
        for i in range(len(self.Y_test)):
            if self.Y_test[i] == self.predict_label[i]:
                count += 1
        print("正确个数：", count)
        accuracy = round(count / len(self.Y_test), 3)
        #print("accuracy:", accuracy)
        #print("need_classification:", self.need_classification)

        return accuracy

    def classification(self, insert_node_id, label):
        # 因为在构建离岸边的时候用的就是str()形式，所以需要此处需要转化为字符串
        adj = [n for n in self.G.neighbors(str(insert_node_id))]  # find the neighbors of the new node
        count0 = 0
        count1 = 0
        count2 = 0
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
        if count0 == len(adj):
            #print("classification_result:", 0)
            self.G.remove_node(str(insert_node_id))
            for n in adj:
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                self.G.add_edge(str(insert_node_id), n)
            self.predict_label.append(1)
        elif count1 == len(adj):
            #print("classification_result:", 1)
            self.G.remove_node(str(insert_node_id))
            for n in adj:
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                self.G.add_edge(str(insert_node_id), n)
            self.predict_label.append(2)
        elif count2 == len(adj):
            #print("classification_result:", 2)
            self.G.remove_node(str(insert_node_id))
            for n in adj:
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                self.G.add_edge(str(insert_node_id), n)
            self.predict_label.append(3)
        else:
            #print("模糊分类情况：")
            #draw_graph(self.G)

            #print(count0, count1, count2)
            dist_list = []
            if count0 >= 0 and count0 < len(adj):
                # delate the edges and node
                if str(insert_node_id) in self.G.nodes():
                    self.G.remove_node(str(insert_node_id))
                # 找到类1中和adj中相同的节点，然后将节点添加到类1中
                node_list = self.G0.nodes()  #这时候还是插入节点之前的G0
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类0中
                for n in neighbor:
                    self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                    self.G.add_edge(str(insert_node_id), n)
                self.get_subgraph() #get the new sungraph to calclulate the measures
                measures0 = calculate_measure(self.G0)  # new subgraph self.G0 measures
                V1, V2 = np.array(self.net0_measure[len(self.net0_measure) - 1]), np.array(measures0)
                euclidean_dist0 = np.linalg.norm(V2 - V1)
                dist_list.append(euclidean_dist0)

            if count1 >= 0 and count1 < len(adj):
                if str(insert_node_id) in self.G.nodes():
                    self.G.remove_node(str(insert_node_id))
                # 找到类1中和adj中相同的节点，
                node_list = self.G1.nodes()
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类1中
                for n in neighbor:
                    self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                    self.G.add_edge(str(insert_node_id), n)
                self.get_subgraph()
                measures1 = calculate_measure(self.G1)
                N1, N2 = np.array(self.net1_measure[len(self.net1_measure) - 1]), np.array(measures1)
                euclidean_dist1 = np.linalg.norm(N2 - N1)
                dist_list.append(euclidean_dist1)

            if count2 >= 0 and count2 < len(adj):
                if str(insert_node_id) in self.G.nodes():
                    self.G.remove_node(str(insert_node_id))
                node_list = self.G2.nodes()
                neighbor = [x for x in node_list if x in adj]
                #添加到2类网络中
                for n in neighbor:
                    self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                    self.G.add_edge(str(insert_node_id), n)
                #self.draw_graph(self.G)
                self.get_subgraph()
                measures2 = calculate_measure(self.G2)
                M1, M2 = np.array(self.net2_measure[len(self.net2_measure) - 1]), np.array(measures2)
                #print("M1, M2:", M1, M2)
                euclidean_dist2 = np.linalg.norm(M2 - M1)
                dist_list.append(euclidean_dist2)
            #确定模糊分类节点的分类标签后，需要将节点移除，然后重新插入到确定的分类标签处
            if str(insert_node_id) in self.G.nodes():
                self.G.remove_node(str(insert_node_id))
            # print(np.array(self.net0_measure), self.net1_measure, self.net2_measure,)
            #print("dist_list:", dist_list)
            # get the classfication ruselt
            list = []
            for x in dist_list:
                if not x == 0:
                    list.append(x)
            min_value = min(list)
            label = int(dist_list.index(min_value)) + 1  #因为label是从1开始
            #print("classification_result:", label)
            self.predict_label.append(label)

            if label == 1:
                node_list = self.G0.nodes()
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类1中
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                for n in neighbor:
                    self.G.add_edge(str(insert_node_id), n)  # add edges

            if label == 2:
                node_list = self.G1.nodes()
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类1中
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                for n in neighbor:
                    self.G.add_edge(str(insert_node_id), n)

            if label == 3:
                node_list = self.G2.nodes()
                neighbor = [x for x in node_list if x in adj]
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                for n in neighbor:
                    self.G.add_edge(str(insert_node_id), n)
            self.need_classification.append(str(insert_node_id))

if __name__ == '__main__':
    df = pd.read_csv('.\Tea.data.csv')
    print(df)

    features = list(df.columns)
    """
    方法一：
    features.remove('class_type')
    features.remove('animal_name')
    print(features)
    """

    features = features[: len(features) - 1]  # 去掉开头和结尾的两列数据

    X = df[features].values.astype(np.float32)
    Y = np.array(df.Class)
    acc0_list = []
    acc1_list = []
    info = []
    fina_result = []

    print(X, Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_test = data_preprocess(X_train, X_test, 2)
    DC = DataClassification(k=2, num_class=3)
    DC.fit(X_train, X_test, Y_train, Y_test)
    accuracy = DC.predict(X_test, Y_test)
    print(accuracy)

    """
    for k in range(2, 6):
        for j in range(5):
            for i in range(5):
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
                X_train, X_test = data_preprocess(X_train, X_test, 2)  # normType=2 归一化到0-1之间
                DC = DataClassification(k=k, num_class=3)
                DC.fit(X_train, X_test, Y_train, Y_test)
                accuracy = DC.predict(X_test, Y_test)
                acc0_list.append(accuracy)

        acc1_list.append(acc0_list)
        acc = np.array(acc1_list)
        print(acc)
        print("k: %d Accuracy: %0.2f (+/- %0.2f)" % (k, acc.mean(), np.std(acc) * 2))
        info.append(k)
        info.append(acc.mean())
        info.append(np.std(acc) * 2)
    fina_result.append(info)
    print(fina_result)
    """
