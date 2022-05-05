import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_moons, make_circles, make_blobs
from sklearn.metrics.pairwise import euclidean_distances, paired_euclidean_distances
from sklearn.preprocessing import Normalizer
import networkx as nx
import math
import pandas as pd
from networkx.algorithms.distance_measures import center as nxCenter
from collections import Counter
from sklearn.model_selection import train_test_split
from GetCOVID_19Data import get_data
import time
class NetworkBaseModel():

    def __init__(self):
        '''

        :param per_class_data_len:  length of per class data
        :param num_classes:         num of classes
        :param k:                   para of kNN
        '''
        self.per_class_data_len = None
        self.train_len = None



        self.train_x = None
        self.data_idxs_list = []
        self.train_y = None

        self.neigh_models = []  #
        self.e_radius = []

        self.G_list = []
        self.mean_dis_list = []
        self.nodes_list = []
        self.edges_list = []
        self.len_list = []  #存储每个组件大小
        self.net_measures = []  # {1:{'averge_degree':[]}}



    def fit(self, x: np.ndarray, y:np.ndarray):
        """

        Args:
            x: array (n, m) 输入数据
            y: (n)
        Returns:

        """
        self.train_x = x
        self.train_y = y


        self.train_len = len(x)


        labels = [i for i in Counter(y)]
        labels.sort()
        self.labels = labels
        #print("self.labels:", self.labels)
        self.num_classes = len(labels)

        "1. build network"
        for ith_class in labels:
            #print("y/ith_classs", type(y), y, ith_class)
            idxs = np.argwhere(y == ith_class).reshape(-1)  #label是按照顺序排的，0， 1， 2， 、、，
                                                            # 所以说从图上通过颜色可以看出来是哪一类
            #print("idxs:", idxs)
            self.data_idxs_list.append(idxs)
            "adjacency matrix"
            #print("x:", x.shape)
            dataset = x[idxs]
            #datatarget = y[idxs]
            #print(dataset)
            data_len = len(dataset)

            #for idx, values in enumerate(dataset):

            #print("ith_class:", ith_class)
            adj_matrix = euclidean_distances(dataset, dataset)

            # 要先求两两平均距离，后面会改动数据。

            mean_dis = np.sum(adj_matrix) / (data_len ** 2 - data_len)

            self.mean_dis_list.append(mean_dis)  #平均差别

            # 把0距离先变成最大999

            adj_matrix[adj_matrix == 0] = 999

            # 先把每一行最小的值设置为1， 即保证每个节点和距离最小的节点有连边

            for idx, item in enumerate(adj_matrix):
                min_idx = np.argmin(item)
                # 因为是对称矩阵
                adj_matrix[idx, min_idx] = 1
                adj_matrix[min_idx, idx] = 1
            # 低于平均距离的建立连边


            adj_matrix[adj_matrix < mean_dis] = 1

            # 将没有连边的部分都设置为0
            adj_matrix[adj_matrix!=1] = 0

            "-- 至此，邻接矩阵生成完毕 --"

            # 构建网络
            self.G_list.append(nx.from_numpy_matrix(adj_matrix))
            #
            # print(self.G_list[ith_class].nodes)
            #print(self.G_list[ith_class].edges)

            "!- merge diff subgraph in a component -!"

            #num_before = nx.number_connected_components(self.G_list[ith_class])
            #print("num_before:", num_before)

            sub_conponents = sorted(nx.connected_components(self.G_list[ith_class]), key=len, reverse=True)

            # print('社区数目',len(sub_conponents))
            center_node = nxCenter(self.G_list[ith_class].subgraph(0))[0]


            #print('---Component----')

            for i in sub_conponents:

                sub_G = self.G_list[ith_class].subgraph(i)

                sub_center_node = nxCenter(sub_G)[0]
                edge = (sub_center_node, center_node)

                self.G_list[ith_class].add_edges_from([edge])
            #num_after = nx.number_connected_components(self.G_list[ith_class])
            #print("num_after:", num_after)


        # 加载各个组件的网络刻度


        for ith_class in range(self.num_classes):
            # res = self.calculate_net_measures(self.G_list[ith_class]).append(self.mean_dis_list[ith_class])
            #res = self.calculate_net_measures(self.G_list[ith_class])
            res = []
            res += [self.mean_dis_list[ith_class]]

            self.net_measures.append(res)

        #print("组件：", self.G_list[0].nodes(), self.G_list[1].nodes())
        self.draw_g()



    def predict(self, x: np.ndarray):

        """

        Args:
            x: numpy (n,m)

        Returns:

        """
        G = nx.disjoint_union_all(self.G_list)  # 合并是哪个组件节点编号和图
        idx = len(G)
        print("G_init:", idx)
        y_pred = []
        print("test_x_len:", len(x))
        count = 0
        for item in x:  # 遍历测试数据
            self.G = nx.disjoint_union_all(self.G_list)  # 合并是哪个组件节点编号和图
            idx = len(self.G)  # 新节点编号

            count += 1
            neighbor = []  #存储新节点与每类网络连边的节点编号
            item = item.reshape(1, -1)
            new_measures = []

            # 分别插入到三个网络当中
            for ith_class in range(self.num_classes):
                #测试插入的组件节点个数
                #print("before:", len(self.G_list[ith_class].nodes()), self.G_list[ith_class].nodes())

                dis_matrix = euclidean_distances(item, self.train_x[self.data_idxs_list[ith_class]])

                aver_dis = np.mean(dis_matrix[0])  # 新节点的平均距离

                min_idx = int(np.argmin(dis_matrix[0]) )         # 最小距离

                edge_idxs = list(np.argwhere(dis_matrix[0] < self.mean_dis_list[ith_class]))
                #print(type(edge_idxs))
                #neighbor.append(edge_idxs)   #新节点邻居
                #print(type(neighbor))

                # 添加节点， 添加连边
                test_node = (idx, {'value': None, 'class': 'test', 'type': 'test'})

                self.G_list[ith_class].add_nodes_from([test_node])

                edges = [(idx, min_idx)]
                for edge_idx in edge_idxs:
                    edges.append((idx, int(edge_idx)))

                self.G_list[ith_class].add_edges_from(edges)
                neighbor.append(edges) #新节点邻居添加进去

                #res = self.calculate_net_measures(self.G_list[ith_class])

                res =[]
                res.append(aver_dis + self.mean_dis_list[ith_class]) #############

                new_measures.append(res)
                #self.draw_g()
                #测试插入后节点个数
                #print("after:", len(self.G_list[ith_class].nodes()), self.G_list[ith_class].nodes())
                #self.draw_g()
                self.G_list[ith_class].remove_node(idx)
                #print("after_remove:", len(self.G_list[ith_class].nodes()), self.G_list[ith_class].nodes())


            diff = paired_euclidean_distances(self.net_measures, new_measures)
            label = int(np.argmin(diff))
            y_pred.append(label)   #返回数组中最小值得索引
            #print("neighbor:", neighbor)
            self.classification(label, idx, neighbor)
            #self.draw_g()

            #检测插入节点后的节点个数
            #G = nx.disjoint_union_all(self.G_list)
            #print("G_after:", len(G.nodes()), G.nodes())


        #print("count:", count)
        #print(self.G_list[0].nodes(), self.G_list[1].nodes(), self.G_list[2].nodes())
        G = nx.disjoint_union_all(self.G_list)
        print("G_final:", len(G.nodes()), G.nodes())
        self.draw_g()
        return np.array(y_pred)

    def classification(self, label, idx, neighbor):
        #print("label:", label)
        #for ith_class in range(self.num_classes):
        #print("neighbor:", len(neighbor), neighbor)
        if label == 0:
            #test_node = (idx, {'value': None, 'class': 'test', 'type': 'test'})
            #self.G_list[0].add_nodes_from([test_node])
            self.G_list[0].add_node(idx, value=None, typeNode="test")
            #edges = []
            #for edge_idx in neighbor[0]:
                #edges.append((test_node, int(edge_idx)))
            self.G_list[0].add_edges_from(neighbor[0])

        if label == 1:
            self.G_list[1].add_node(idx, value=None, typeNode="test")
            self.G_list[1].add_edges_from(neighbor[1])

        if label == 2:
            self.G_list[2].add_node(idx, value=None, typeNode="test")
            self.G_list[2].add_edges_from(neighbor[2])

        if label == 3:
            self.G_list[3].add_node(idx, value=None, typeNode="test")
            self.G_list[3].add_edges_from(neighbor[3])

        if label == 4:
            self.G_list[4].add_node(idx, value=None, typeNode="test")
            self.G_list[4].add_edges_from(neighbor[4])

        if label == 5:
            self.G_list[5].add_node(idx, value=None, typeNode="test")
            self.G_list[5].add_edges_from(neighbor[5])

        if label == 6:
            self.G_list[6].add_node(idx, value=None, typeNode="test")
            self.G_list[6].add_edges_from(neighbor[6])

        if label == 7:
            self.G_list[7].add_node(idx, value=None, typeNode="test")
            self.G_list[7].add_edges_from(neighbor[7])

        if label == 8:
            self.G_list[8].add_node(idx, value=None, typeNode="test")
            self.G_list[8].add_edges_from(neighbor[8])

        if label == 9:
            self.G_list[9].add_node(idx, value=None, typeNode="test")
            self.G_list[9].add_edges_from(neighbor[9])



    def check(self,x,y):
        y_hat = self.predict(x)
        print("origanl_y:", y)
        print("predict:", y_hat)
        acc = np.sum(y_hat == y) / len(y)
        return acc

    """
    def calculate_net_measures(self, net):


        degree_assortativity = nx.degree_assortativity_coefficient(G=net)

        #average_clustering_coefficient = nx.average_clustering(G=net)
        #average_degree = np.mean([i[1] for i in nx.degree(net)])  # nx.degree 返回的是每个节点的度, 所以要获取再求平均
        #dimameter = nx.algorithms.distance_measures.diameter(net)

        return [degree_assortativity, average_clustering_coefficient, average_degree, dimameter]
    

    def generate_net_measures(self):
        seg_point = self.per_class_data_len + 1
        for per_class in range(self.num_classes):
            nodes = list(self.G.nodes)[seg_point * per_class:seg_point * (per_class + 1)]
            sub_G = self.G.subgraph(nodes=nodes)
            self.net_measures[per_class] = self.calculate_net_measures(sub_G)
    """


    def draw_g(self):
        color_map = {0: 'red', 1: 'green', 2: 'yellow', 3: 'b', 4: 'm', 5: 'c', 6: 'black',
                    7: 'grey', 8: 'y', 9: 'magenta'}
        plt.figure("Graph", figsize=(12, 12))
        G = nx.disjoint_union_all(self.G_list)
        color_list = []
        for idx, thisG in enumerate(self.G_list):
            color_list += [color_map[idx]] * len(thisG.nodes)

        """
        for index, (node, typeNode) in enumerate(G.nodes(data='typeNode')):
            if (typeNode == 'test'):
                color_list[index] = 'purple'
        """

        """
        for node in G.nodes():
            if G.nodes[node]:
                color_list[node] = 'purple'
            if G._node[node]]["type"]:
                if G._node[node]]["type"] == "test":
                    color_list[node] = "purple"
        """
        pos = nx.spring_layout(G)

        nx.draw_networkx(G, pos, with_labels=True, node_size=200, node_color=color_list)  #

        plt.show()
        """
        #自己写的代码，
        node_list = [[], [], []]
        #print(len(node_list))
        for i in range(self.num_classes): #得到每个组件长度
            self.len_list.append(len(self.G_list[i].nodes()))
        nodes = []
        #print(type(nodes))
        n = 0
        m = self.len_list[0]
        for a in range(self.num_classes):
            for i in range(n, m):
                node_list[a].append(i)
            if a < self.num_classes - 1:
                n += self.len_list[a]
                m += self.len_list[a+1]

        #print("node_list:", np.array(node_list))
        #print(self.len_list)

        print(len(G.nodes()), G.nodes())
        #nodes_l = [ i for b in range(self.num_classes) for i in node_list[b]]
        #print(nodes_l)
        #color_list = [color_map[G.nodes[node]['label']] for node in G.nodes()]

        nx.draw_networkx_nodes(G, pos, nodelist=node_list[0], node_color="g", node_size=300)  #
        nx.draw_networkx_nodes(G, pos, nodelist=node_list[1], node_color="r", node_size=300)  #
        nx.draw_networkx_nodes(G, pos, nodelist=node_list[2], node_color="y", node_size=300)  #
        nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
        nx.draw_networkx_edges(G, pos)

        plt.show()
        """



if __name__ == '__main__':

    start = time.time()
    data = load_iris() #iris数据集
    #data = load_wine() #wine数据集
    #data = load_breast_cancer()
    x = data['data']
    y = data['target']

    #covid-19数据集
    #x, y = get_data()  #covid-19数据集 label：0：normal， 1：P， 2：covid-19
    #print(x.shape, len(y))
    #df = pd.read_csv(r"C:\Users\Yan\Desktop\CovidData_n12.csv")
    #df = pd.read_csv(r"C:\Users\Yan\Desktop\Brazil study files\Dataset\Pima.csv")

    """
    features = list(df.columns)
    features = features[: len(features) - 1]  # 去掉开头和结尾的两列数据
    #print(features)
    x = df[features].values.astype(np.float32)
    y = np.array(df.Outcome)
    """

    a = []
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
        model = NetworkBaseModel()
        model.fit(x_train, y_train)
        acc = model.check(x_test, y_test)
        print("acc:", acc)
        a.append(acc)

    print(a)
    mean_acc = np.mean(a)
    max = np.max(a)
    min = np.min(a)
    var = np.var(a)
    print(a)
    print("%f +- %f", (mean_acc, var))
    print(min, max)

    end = time.time()
    print("time:", end - start)