import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import pandas as pd

class DataClassification(object):
    """iris数据集分类"""

    def __init__(self, k, num_class):
        self.k = k
        # self.g = []
        self.train_data, self.train_target, self.X_test, self.Y_test = self.get_data()
        self.data_len = len(self.train_data)  # 此程序是24
        self.num_class = num_class
        self.per_class_data_len = int(self.data_len / self.num_class)
        #self.plot_data(self.train_data)  # 直接执行此函数
        self.nbrs = []  # 用来存储是哪个类别网络的nbrs
        self.radius = []  # 用来存储是哪个类别的
        self.weight_alpha = [0.1, 0.4, 0.4]  # measures权重
        self.color_map = {1: 'red', 2: 'green', 3: 'yellow', 5: 'b', 6: 'm', 7: 'c'}
        self.net0_measure = []  # 存储每一类别插入Y_items中的数据时的指标
        self.impact0 = []  # storage the impact of the first class
        self.net1_measure = []
        self.impact1 = []
        self.net2_measure = []
        self.net3_measure = []
        self.net4_measure = []
        self.net5_measure = []
        self.net6_measure = []
        self.G = nx.Graph()

        self.predict_label = []

        # 初始化运行程序，一开始就运行
        self.build_init_network(label=True)
        self.need_classification = []  # 计算模糊分类节点次数
        self.single_node_insert()
        #self.accuracy()

    def get_data(self):
        """获取数据集"""
        df = pd.read_csv('glass.csv')
        features = list(df.columns)
        """
        方法一：
        features.remove('class_type')
        features.remove('animal_name')
        print(features)
        """
        features = features[1: len(features) - 1]  # 去掉开头和结尾的两列数据
        #print(features)
        X = df[features].values.astype(np.float32)
        Y = np.array(df.Type)

        train_data, X_test, train_target,  Y_test = train_test_split(X, Y, test_size=0.2)
        train_data, X_test = self.data_preprocess(train_data, X_test)

        print("总的数据集:")
        print(X, Y)
        #print(iris_data[85])

        print("X_traing:")
        print(train_data, train_target)
        #print(train_data[85])
        #print(iris_data.index(train_data[85]))
        print(len(train_data))
        #print(train_data[85], train_data[84])


        print("X_test:")

        print(np.array(X_test), np.array(Y_test))
        print(len(X_test))

        print(len(train_data))
        print(len(X_test))

        return train_data, train_target, X_test, Y_test


    def data_preprocess(self, data_train, data_test):
        """特征工程（归一化）"""
        # 归一化
        scaler = preprocessing.MinMaxScaler().fit(data_train)
        train_data = scaler.transform(data_train)
        test_data = scaler.transform(data_test)

        return train_data, test_data

    def KNN(self, nbrs, train_data):
        """
        KNN获取节点邻居和邻居索引
        :param nbrs:
        :param train_data:
        """
        distances, indices = nbrs.kneighbors(train_data)
        return distances, indices

    def get_radius(self, distances):

        return np.median(distances)  # 中位数

    def epsilon_radius(self, nbrs, train_data, radius):
        """

        :param nbrs:
        :param train_data:
        :param radius:
        :return:
        """
        nbrs.set_params(radius=radius)
        nbrs_distances, nbrs_indices = nbrs.radius_neighbors(train_data)

        return nbrs_distances, nbrs_indices

    def calculate_measure(self, G):
        """
        :param net: 构建的网络g
        :param nodes: 每一类的网络节点
        :return:
        """
        measures = []

        # 平均度，平均最短路径，平均聚类系数， 同配性， 传递性
        # 1.  平均度
        ave_deg = G.number_of_edges() * 2 / G.number_of_nodes()
        ave_deg = round(ave_deg, 3)
        # print("平均度为：%f" % ave_deg)
        measures.append(ave_deg)


        # 2.  平均最短路径长度(需要图是连通的)
        ave_shorest = nx.average_shortest_path_length(G)
        ave_shorest = round(ave_shorest, 3)
        #print("平均最短路径：", ave_shorest)
        #measures.append(ave_shorest)


        # 3.  平均聚类系数
        ave_cluster = nx.average_clustering(G)
        ave_cluster = round(ave_cluster, 3)
        # print("平均聚类系数：%f" % ave_cluster)
        measures.append(ave_cluster)


        """
        #4.  度同配系数 Compute degree assortativity of graph
        assortativity = nx.degree_assortativity_coefficient(G)
        assortativity = round(assortativity, 3)
        # print("同配性：%f" % assortativity)
        measures.append(assortativity)
        """

        # 5.  传递性transitivity：计算图的传递性，g中所有可能三角形的分数。
        tran = nx.transitivity(G)
        tran = round(tran, 3)
        #print("三角形分数：%f" % tran)
        measures.append(tran)

        return measures

    def build_init_network(self, label):
        # current_data = self.train_data[self.per_class_data_len * i:self.per_class_data_len * (i + 1), :]
        # print(current_data)
        for index, instance in enumerate(self.train_data):
            self.G.add_node(str(index), value=instance, typeNode="init_net", label=self.train_target[index])
        # print(self.nodes_list)

        # 切片范围必须是整型
        temp_nbrs = NearestNeighbors(self.k, metric='euclidean')
        temp_nbrs.fit(self.train_data)
        self.nbrs.append(temp_nbrs)  # 将每一类的nbrs都添加进列表， 这个
        knn_distances, knn_indices = self.KNN(temp_nbrs, self.train_data)
        # print(" ")
        temp_radius = self.get_radius(knn_distances)

        self.radius.append(temp_radius)  # 将每一类的radius都添加进radius
        # print("temp_radius", self.radius)
        radius_distances, radius_indices = self.epsilon_radius(temp_nbrs, self.train_data, temp_radius)
        # print(knn_indices, np.array(radius_indices))
        # print(" ")
        # print(np.array(radius_indices))
        # 添加连边
        for idx, one_data in enumerate(self.train_data):  # 这个语句仅仅是获取索引indx，然后给他连边
            # print(knn_indices[idx], radius_indices[idx])
            if (len(radius_indices[idx])) > self.k:  # 判断用哪种方法连边构图,因为,radius应用于稠密区域，邻居大于k的话就是稠密
                print("radius technique:")
                print(idx, radius_indices[idx], knn_indices[idx])

                for indiceNode, indicesNode in enumerate(radius_indices):
                    for tmpi, indice in enumerate(indicesNode):
                        if (str(indice) == str(indiceNode)):
                            continue
                        if (self.G.nodes()[str(indice)]["label"] == self.G.nodes()[str(indiceNode)][
                            "label"] or not label):
                            self.G.add_edge(str(indice), str(indiceNode), weight=radius_distances[indiceNode][tmpi])

            else:
                print("KNN technique:")
                print(idx, knn_indices[idx], radius_indices[idx])
                for indiceNode, indicesNode in enumerate(knn_indices):
                    for tmpi, indice in enumerate(indicesNode):
                        if (str(indice) == str(indiceNode)):
                            continue
                        if (self.G.nodes()[str(indice)]["label"] == self.G.nodes()[str(indiceNode)][
                            "label"] or not label):
                            self.G.add_edge(str(indice), str(indiceNode), weight=knn_distances[indiceNode][tmpi])

            # do the next steps when there are thingle node (small components)

            """
            num = nx.number_connected_components(self.G)
            if num > self.num_class:
                Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)
                list0 = [list(n) for n in Gcc] #components list
                print("components_list:", list0)
                for i in range(len(list0)):
                    Gi = nx.Graph(list0[i])
            """

            # 单节点情况，和同类最近编号节点连边三个
        for idx, one_data in enumerate(self.train_data):
            count = 0
            new_id = str(idx)  # 因为neighbors迭代器用的节点编号的字符串，所以需要转化为字符串
            # print(self.G.nodes())
            adj = [n for n in self.G.neighbors(new_id)]  # find the neighbors of the new node
            if len(adj) == 0:
                count += 1
                print("sing_node_num:", count)
                print("idx:", idx)
                count = 0
                for i in range(len(self.G.nodes())):
                    if self.G._node[str(idx)]["label"] == self.G._node[str(i)]["label"]:
                        self.G.add_edges_from([(str(idx), str(i))])
                        count += 1
                    if count == self.k:
                        break
            #print("")

        print("nodes:", self.G.nodes())
        self.draw_graph(self.G)
        self.get_subgraph()
        self.draw_graph(self.G)
        self.get_subgraph()
        print("self.G0 info:")
        self.draw_graph(self.G0)
        for n in self.G0.nodes():
            if n in self.G0.nodes():
                label = self.G._node[n]["label"]
        print("self.G0_label:", label)

        print("self.G1 info:")
        self.draw_graph(self.G1)
        for n in self.G1.nodes():
            if n in self.G1.nodes():
                label = self.G1._node[n]["label"]
        print("self.G1_label:", label)

        print("self.G2 info:")
        self.draw_graph(self.G2)
        for n in self.G2.nodes():
            if n in self.G2.nodes():
                label = self.G2._node[n]["label"]
        print("label:", label)

    def draw_graph(self, G):
        plt.figure("Graph", figsize=(9, 9))
        pos = nx.spring_layout(G)
        color_list = [self.color_map[G.nodes[node]['label']] for node in G.nodes()]

        for index, (node, typeNode) in enumerate(G.nodes(data='typeNode')):
            if (typeNode == 'test'):
                color_list[index] = 'purple'
                # color_list = [self.color_map[G.nodes[node]['label']] for node in G.nodes()]
        nx.draw_networkx(G, pos, node_color=color_list, with_labels=True,
                         node_size=100, font_size=6)  # 节点默认大小为300, 节点标签大小默认为12
        plt.show()
        print("node_num：", len(G.nodes()))
        print("edges_num：", len(G.edges()))
        print(" ")

    def get_subgraph(self):
        """得到子图，并画出来"""
        num = nx.number_connected_components(self.G)
        print("num_components:", num)
        Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)
        list = []
        for n in range(num):
            G = self.G.subgraph(Gcc[n])
            count0 = count1 = count2 = count3 = count4 = count5 = count6 = 0
            for m in G.nodes():
                if G._node[m]["label"] == 1:
                    count0 += 1
                    if count0 >= 5:  #设置阈值，查找大的组件构建初试类网络，用以后边吞并小的组件
                        self.G0 = G
                if G._node[m]["label"] == 2:
                    count1 += 1
                    if count1 >= 2:
                        self.G1 = G
                if G._node[m]["label"] == 3:
                    count2 += 1
                    if count2 >= 2:
                        self.G2 = G
                if G._node[m]["label"] == 4:
                    count3 += 1
                    if count3 >= 3:
                        self.G3 = G
                if G._node[m]["label"] == 5:
                    count4 += 1
                    if count4 >= 2:
                        self.G4 = G
                if G._node[m]["label"] == 6:
                    count5 += 1
                    if count5 >= 2:
                        self.G5 = G
                if G._node[m]["label"] == 7:
                    count6 += 1
                    if count6 >= 2:
                        self.G6 = G

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
                    if G._node[n]["label"] == 4:
                        count = 0
                        for a in self.G3.nodes():
                            if n == a:
                                continue
                            self.G.add_edge(str(n), str(a))
                            count += 1
                            if count == self.k:
                                break
                    if G._node[n]["label"] == 5:
                        count = 0
                        for a in self.G4.nodes():
                            if n == a:
                                continue
                            self.G.add_edge(str(n), str(a))
                            count += 1
                            if count == self.k:
                                break
                    if G._node[n]["label"] == 6:
                        count = 0
                        for a in self.G5.nodes():
                            if n == a:
                                continue
                            self.G.add_edge(str(n), str(a))
                            count += 1
                            if count == self.k:
                                break
                    if G._node[n]["label"] == 7:
                        count = 0
                        for a in self.G6.nodes():
                            if n == a:
                                continue
                            self.G.add_edge(str(n), str(a))
                            count += 1
                            if count == self.k:
                                break

    def single_node_insert(self):
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
        # 添加节点
        for index, instance in enumerate(self.X_test):
            # label = 4
            label = self.Y_test[index]
            print("label:", label)
            if (not len(self.Y_test) == 0):
                label = self.Y_test[index]
                # print(index, instance)
            # 计算插入节点之前的各个类别网络的measures
            self.get_subgraph()
            # self.draw_graph(self.G0)
            measures0 = self.calculate_measure(self.G0)
            self.net0_measure.append(measures0)
            # self.draw_graph(self.G1)
            measures1 = self.calculate_measure(self.G1)
            self.net1_measure.append(measures1)
            # self.draw_graph(self.G2)
            measures2 = self.calculate_measure(self.G2)
            self.net2_measure.append(measures2)
            measures3 = self.calculate_measure(self.G3)
            self.net3_measure.append(measures3)
            measures4 = self.calculate_measure(self.G4)
            self.net4_measure.append(measures4)
            measures5 = self.calculate_measure(self.G5)
            self.net5_measure.append(measures5)
            measures6 = self.calculate_measure(self.G6)
            self.net6_measure.append(measures6)

            # 插入新的节点构建连边
            insert_node_id = len(list(self.G.nodes()))
            print("insert_node_id:", insert_node_id)
            self.G.add_node(str(insert_node_id), values=instance, typeNode='test', label=label)
            # print(len(self.g.nodes()))

            # print(self.nbrs, self.radius)
            radius_distances, radius_indices = self.epsilon_radius(self.nbrs[0], [instance],
                                                                   self.radius[0])
            # print("radius:", radius_indices)
            distances, indices = self.KNN(self.nbrs[0], [instance])
            # print(distances, indices)    #所以有有可能包含自身 ，到时候添加边要过滤掉

            # 添加到训练网络中
            for idx, one_data in enumerate(self.train_data):  # 这个语句仅仅是获取索引indx，然后给他连边
                # print(radius_indices[idx])
                if (len(radius_indices)) > self.k:  # 判断用哪种方法连边构图,因为,radius应用于稠密区域，邻居大于k的话就是稠密
                    # print(radius_indices[idx])
                    for indiceNode, indicesNode in enumerate(radius_indices):
                        for tmpi, indice in enumerate(indicesNode):
                            if (str(indice) == str(indiceNode)):
                                continue
                            self.G.add_edge(str(indice), str(insert_node_id), weight=radius_distances[indiceNode][tmpi])
                else:
                    for indiceNode, indicesNode in enumerate(indices):
                        for tmpi, indice in enumerate(indicesNode):
                            if (str(indice) == str(indiceNode)):
                                continue
                            self.G.add_edge(str(indice), str(insert_node_id), weight=distances[indiceNode][tmpi])
            self.classification(insert_node_id, int(self.Y_test[index]))
            print("=" * 100)

        print("finally_G_info:")
        self.draw_graph(self.G)
        #self.get_subgraph()
        #print("finally_subgraph_info:")
        #self.draw_graph(self.G0)
        #self.draw_graph(self.G1)
        #self.draw_graph(self.G2)

    def classification(self, insert_node_id, label):
        # 因为在构建离岸边的时候用的就是str()形式，所以需要此处需要转化为字符串
        """
        print(self.G.nodes())
        print(self.G.edges())
        print(self.G0.nodes())
        print(self.G1.nodes())
        print(self.G2.nodes())
        """
        adj = [n for n in self.G.neighbors(str(insert_node_id))]  # find the neighbors of the new node
        print("adj:", adj)
        # check which class the link of the new node belongs to
        count0 = count1 = count2 = count3 = count4 = count5 = count6 = 0
        for n in adj:
            if n in self.G0.nodes():
                label = self.G._node[n]["label"]
                print("label:", label)
                count0 += 1
            elif n in self.G1.nodes():
                label = self.G._node[n]["label"]
                print("label:", label)
                count1 += 1
            elif n in self.G2.nodes():
                label = self.G._node[n]["label"]
                print("label:", label)
                count2 += 1
            elif n in self.G3.nodes():
                label = self.G._node[n]["label"]
                print("label:", label)
                count3 += 1
            elif n in self.G4.nodes():
                label = self.G._node[n]["label"]
                print("label:", label)
                count4 += 1
            elif n in self.G5.nodes():
                label = self.G._node[n]["label"]
                print("label:", label)
                count5 += 1
            elif n in self.G6.nodes():
                label = self.G._node[n]["label"]
                print("label:", label)
                count6 += 1

        print(count0, count1, count2, count3, count4, count5, count6,)
        if count0 == len(adj):
            print("classification_result:", 1)
            self.G.remove_node(str(insert_node_id))
            for n in adj:
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                self.G.add_edge(str(insert_node_id), n)
            self.predict_label.append(1)
        elif count1 == len(adj):
            print("classification_result:", 2)
            self.G.remove_node(str(insert_node_id))
            for n in adj:
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                self.G.add_edge(str(insert_node_id), n)
            self.predict_label.append(2)
        elif count2 == len(adj):
            print("classification_result:", 3)
            self.G.remove_node(str(insert_node_id))
            for n in adj:
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                self.G.add_edge(str(insert_node_id), n)
            self.predict_label.append(3)
        elif count3 == len(adj):
            print("classification_result:", 4)
            self.G.remove_node(str(insert_node_id))
            for n in adj:
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                self.G.add_edge(str(insert_node_id), n)
            self.predict_label.append(4)
        elif count4 == len(adj):
            print("classification_result:", 5)
            self.G.remove_node(str(insert_node_id))
            for n in adj:
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                self.G.add_edge(str(insert_node_id), n)
            self.predict_label.append(5)
        elif count5 == len(adj):
            print("classification_result:", 6)
            self.G.remove_node(str(insert_node_id))
            for n in adj:
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                self.G.add_edge(str(insert_node_id), n)
            self.predict_label.append(6)
        elif count6 == len(adj):
            print("classification_result:", 7)
            self.G.remove_node(str(insert_node_id))
            for n in adj:
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                self.G.add_edge(str(insert_node_id), n)
            self.predict_label.append(7)
        else:
            print("模糊分类情况：")
            self.draw_graph(self.G)
            #self.G.remove_node(new_id)
            """
            self.get_subgraph()
            self.draw_graph(self.G0)
            self.draw_graph(self.G1)
            self.draw_graph(self.G2)
            """
            print(count0, count1, count2, count3, count4, count5, count6)
            dist_list = []
            # if count0 == 0:
            #    dist_list[]

            if count0 >= 0 and count0 < len(adj):
                # delate the edges and node
                if str(insert_node_id) in self.G.nodes():
                    self.G.remove_node(str(insert_node_id))
                # 找到类1中和adj中相同的节点，然后将节点添加到类1中
                #self.G.remove_node(new_id)
                node_list = self.G0.nodes()  #这时候还是插入节点之前的G0
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类0中
                for n in neighbor:
                    self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                    self.G.add_edge(str(insert_node_id), n)
                self.draw_graph(self.G)  #暂时分类的新图
                self.get_subgraph() #get the new sungraph to calclulate the measures


                measures0 = self.calculate_measure(self.G0)  # new subgraph self.G0 measures
                V1, V2 = np.array(self.net0_measure[len(self.net0_measure) - 1]), np.array(measures0)
                print("v1, v2:", V1, V2)
                euclidean_dist0 = np.linalg.norm(V2 - V1)
                dist_list.append(euclidean_dist0)

                #self.G.remove_node(str(insert_node_id))

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
                self.draw_graph(self.G)
                self.get_subgraph()


                measures1 = self.calculate_measure(self.G1)
                N1, N2 = np.array(self.net1_measure[len(self.net1_measure) - 1]), np.array(measures1)
                print("N1, N2:", N1, N2)
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
                self.draw_graph(self.G)
                self.get_subgraph()


                measures2 = self.calculate_measure(self.G2)
                M1, M2 = np.array(self.net2_measure[len(self.net2_measure) - 1]), np.array(measures2)
                print("M1, M2:", M1, M2)
                euclidean_dist2 = np.linalg.norm(M2 - M1)
                dist_list.append(euclidean_dist2)
            if count3 >= 0 and count3 < len(adj):
                if str(insert_node_id) in self.G.nodes():
                    self.G.remove_node(str(insert_node_id))
                node_list = self.G3.nodes()
                neighbor = [x for x in node_list if x in adj]
                #添加到2类网络中

                for n in neighbor:
                    self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                    self.G.add_edge(str(insert_node_id), n)
                self.draw_graph(self.G)
                self.get_subgraph()

                measures3 = self.calculate_measure(self.G3)
                M1, M3 = np.array(self.net3_measure[len(self.net3_measure) - 1]), np.array(measures3)
                print("M1, M3:", M1, M3)
                euclidean_dist3 = np.linalg.norm(M3 - M1)
                dist_list.append(euclidean_dist3)
            if count4 >= 0 and count4 < len(adj):
                if str(insert_node_id) in self.G.nodes():
                    self.G.remove_node(str(insert_node_id))
                node_list = self.G4.nodes()
                neighbor = [x for x in node_list if x in adj]
                #添加到2类网络中

                for n in neighbor:
                    self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                    self.G.add_edge(str(insert_node_id), n)
                self.draw_graph(self.G)
                self.get_subgraph()


                measures4 = self.calculate_measure(self.G4)
                M1, M4 = np.array(self.net4_measure[len(self.net4_measure) - 1]), np.array(measures4)
                print("M1, M4:", M1, M4)
                euclidean_dist4 = np.linalg.norm(M4 - M1)
                dist_list.append(euclidean_dist4)
            if count5 >= 0 and count5 < len(adj):
                if str(insert_node_id) in self.G.nodes():
                    self.G.remove_node(str(insert_node_id))
                node_list = self.G5.nodes()
                neighbor = [x for x in node_list if x in adj]
                #添加到2类网络中

                for n in neighbor:
                    self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                    self.G.add_edge(str(insert_node_id), n)
                self.draw_graph(self.G)
                self.get_subgraph()


                measures5 = self.calculate_measure(self.G5)
                M1, M5 = np.array(self.net5_measure[len(self.net5_measure) - 1]), np.array(measures5)
                print("M1, M5:", M1, M5)
                euclidean_dist5 = np.linalg.norm(M5 - M1)
                dist_list.append(euclidean_dist5)
            if count6 >= 0 and count6 < len(adj):
                if str(insert_node_id) in self.G.nodes():
                    self.G.remove_node(str(insert_node_id))
                node_list = self.G6.nodes()
                neighbor = [x for x in node_list if x in adj]
                #添加到2类网络中

                for n in neighbor:
                    self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                    self.G.add_edge(str(insert_node_id), n)
                self.draw_graph(self.G)
                self.get_subgraph()


                measures6 = self.calculate_measure(self.G6)
                M1, M6 = np.array(self.net6_measure[len(self.net6_measure) - 1]), np.array(measures6)
                print("M1, M6:", M1, M6)
                euclidean_dist6 = np.linalg.norm(M6 - M1)
                dist_list.append(euclidean_dist6)

            #确定模糊分类节点的分类标签后，需要将节点移除，然后重新插入到确定的分类标签处
            if str(insert_node_id) in self.G.nodes():
                self.G.remove_node(str(insert_node_id))
            # print(np.array(self.net0_measure), self.net1_measure, self.net2_measure,)
            print("dist_list:", dist_list)
            # get the classfication ruselt
            #因为有些measures是0，左移不能取最小值得索引，所以建立新列表找到新列表中的非零的最小元素
            #然后再原列中找到这个元素对应的下标值
            list = []
            for x in dist_list:
                if not x == 0:
                    list.append(x)
            min_value = min(list)
            label = int(dist_list.index(min_value))+1 #因为这个数据集的target是从1
            print("classification_result:", label)
            self.predict_label.append(label)

            if label == 0:
                node_list = self.G0.nodes()
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类1中
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                for n in neighbor:
                    self.G.add_edge(str(insert_node_id), n)  # add edges

            if label == 1:
                node_list = self.G1.nodes()
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类1中
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                for n in neighbor:
                    self.G.add_edge(str(insert_node_id), n)

            if label == 2:
                node_list = self.G2.nodes()
                neighbor = [x for x in node_list if x in adj]
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                for n in neighbor:
                    self.G.add_edge(str(insert_node_id), n)

            if label == 3:
                node_list = self.G3.nodes()
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类1中
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                for n in neighbor:
                    self.G.add_edge(str(insert_node_id), n)  # add edges
            if label == 4:
                node_list = self.G4.nodes()
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类1中
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                for n in neighbor:
                    self.G.add_edge(str(insert_node_id), n)  # add edges

            if label == 5:
                node_list = self.G5.nodes()
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类1中
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                for n in neighbor:
                    self.G.add_edge(str(insert_node_id), n)  # add edges
            if label == 6:
                node_list = self.G6.nodes()
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类1中
                self.G.add_node(str(insert_node_id), typeNode='test', label=label)
                for n in neighbor:
                    self.G.add_edge(str(insert_node_id), n)  # add edges
            self.need_classification.append(str(insert_node_id))

    def accuracy(self):
        label = list(map(int, self.Y_test))  # 廖雪峰，高阶函数内容
        print("original_label:", label)
        print("predict_label :", self.predict_label)

        count = 0
        for i in range(len(self.Y_test)):
            if self.Y_test[i] == self.predict_label[i]:
                count += 1
        print("正确个数：", count)
        accuracy = round(count / len(self.Y_test), 3)
        print("accuracy:", accuracy)
        print("need_classification:", self.need_classification)

        return accuracy
if __name__ == '__main__':
    DC = DataClassification(2, 7)
    DC.accuracy()
