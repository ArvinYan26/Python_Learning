import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_moons, make_circles, make_blobs, load_digits
from sklearn.metrics.pairwise import euclidean_distances, paired_euclidean_distances
from sklearn.preprocessing import Normalizer
import networkx as nx
import math
import pandas as pd
from networkx.algorithms.distance_measures import center as nxCenter
from collections import Counter
from sklearn.model_selection import train_test_split
#from GetCOVID_19Data import get_data
from GetCOVID_19Data1 import get_data  #原图像傅里叶变换，两类（正常和新冠）
from sklearn import preprocessing
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns

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

    def get_params(self, in_rate, c_rate):
        """

        :param in_rate:init_threshold_rate
        :param c_rate: classicfication_threshold_rate
        :return:
        """
        #return {'k': self.k, 'num_class': self.class_num}
        self.init_rate = in_rate
        self.c_r = c_rate

    def data_preprcess(self, data):
        min_max_scaler = preprocessing.MinMaxScaler().fit(data)
        new_data = min_max_scaler.transform(data)
        return new_data

    def fit(self, x: np.ndarray, y:np.ndarray):
        """

        Args:
            x: array (n, m) 输入数据
            y: (n)
        Returns: predict_label

        """

        self.train_x = x
        self.train_y = y
        self.train_len = len(x)

        labels = [i for i in Counter(y)]
        labels.sort()
        self.labels = labels
        #print("self.labels:", self.labels)
        self.num_classes = len(labels)
        self.data = []
        "1. build network"
        for ith_class in labels:
            # label是按照顺序排的，0， 1， 2， 、, # 所以说从图上通过颜色可以看出来是哪一类
            idxs = np.argwhere(y == ith_class).reshape(-1)
            self.data_idxs_list.append(idxs)
            "adjacency matrix"
            dataset = x[idxs]
            self.data.append(dataset)

            data_len = len(dataset)
            adj_matrix = euclidean_distances(dataset, dataset)
            # 要先求两两平均距离，后面会改动数据。
            mean_dis = np.sum(adj_matrix) / (data_len ** 2 - data_len)
            #print("mean_dis:", mean_dis)
            #mean_dis = mean_dis*self.t #变化阈值
            self.mean_dis_list.append(mean_dis)  #平均差别

        print("self.mean_dis_list:", self.mean_dis_list)
        for ith_class in labels:
            #data_len = len(self.data[ith_class])
            adj_matrix = euclidean_distances(self.data[ith_class], self.data[ith_class])
            # 把0距离先变成最大999
            adj_matrix[adj_matrix == 0] = 999

            # 先把每一行最小的值设置为1， 即保证每个节点和距离最小的节点有连边

            for idx, item in enumerate(adj_matrix):
                min_idx = np.argmin(item)
                # 因为是对称矩阵
                adj_matrix[idx, min_idx] = 1
                adj_matrix[min_idx, idx] = 1
            # 低于平均距离的建立连边


            #adj_matrix[adj_matrix < mean_dis] = 1
            #选固定值，将差距拉到最大
            adj_matrix[adj_matrix < np.median(self.mean_dis_list)*self.init_rate] = 1

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

            for i in sub_conponents: #合并节点就是每个子图中中心节点连接即可

                sub_G = self.G_list[ith_class].subgraph(i)

                sub_center_node = nxCenter(sub_G)[0]
                edge = (sub_center_node, center_node)

                self.G_list[ith_class].add_edges_from([edge])
            #num_after = nx.number_connected_components(self.G_list[ith_class])
            #print("num_after:", num_after)


        # 加载各个组件的网络刻度

        print("self.G_list[1]:", len(self.G_list[1].nodes()))
        for ith_class in range(self.num_classes):
            # res = self.calculate_net_measures(self.G_list[ith_class]).append(self.mean_dis_list[ith_class])
            res = self.calculate_net_measures(self.G_list[ith_class], [])
            #res = []
            #res += [self.mean_dis_list[ith_class]]

            self.net_measures.append(res)

        #print("组件：", self.G_list[0].nodes(), self.G_list[1].nodes())
        print("self.net_measures:", np.array(self.net_measures))
        self.draw_g()



    def predict(self, x: np.ndarray, y):

        """

        Args:
            x: test_data
        Returns:

        """
        G = nx.disjoint_union_all(self.G_list)  # 合并是哪个组件节点编号和图
        idx = len(G)
        #print("G_init:", idx)
        y_pred = []
        print("test_x_len:", len(x))
        count = 0
        #x = self.data_preprcess(x)
        for idx, item in enumerate(x):  # 遍历测试数据

            l = y[idx]
            print("label:", l)

            self.G = nx.disjoint_union_all(self.G_list)  # 合并是哪个组件节点编号和图
            idx = len(self.G)  # 新节点编号

            count += 1
            neighbor = []  #存储新节点与每类网络连边的节点编号
            item = item.reshape(1, -1)
            new_measures = []

            # 分别插入到三个网络当中
            for ith_class in range(self.num_classes):
                #print(len(self.data_idxs_list), self.data_idxs_list[ith_class])
                #print(self.train_x[self.data_idxs_list[ith_class]])
                #测试插入的组件节点个数
                #print("before:", len(self.G_list[ith_class].nodes()), self.G_list[ith_class].nodes())
                #计算新数据与初始化网络之间的每个数据间的距离（待定，并非新网络中两两数据之间的距离）
                #print(item.shape, self.train_x[self.data_idxs_list[ith_class]].shape)
                dis_matrix = euclidean_distances(item, self.train_x[self.data_idxs_list[ith_class]])
                #print("dis_matrix:", dis_matrix)
                #dis_matrix = self.data_preprcess(dis_matrix)
                #print("after_dis_matrix:", dis_matrix)
                """
                #将新数据与原始网络数据合并计算平均距离
                new_data = np.vstack((self.train_x[self.data_idxs_list[ith_class]], item))
                adj_matrix = euclidean_distances(new_data, new_data)
                adj_matrix = self.data_preprcess(adj_matrix)
                # print("adj_matrix:", adj_matrix)
                # 要先求两两平均距离，后面会改动数据。
                mean_dis = np.sum(adj_matrix) / (len(adj_matrix) ** 2 - len(adj_matrix))
                """

                aver_dis = np.mean(dis_matrix[0])  # 新节点与初始网络中每个数据间的距离的平均距离
                #aver_dis = aver_dis*self.t
                #print("aver_dis:", aver_dis)

                min_idx = int(np.argmin(dis_matrix[0]))         # 最小距离

                #edge_idxs = list(np.argwhere(dis_matrix[0] < self.mean_dis_list[ith_class]))
                #三个网络的最小平均值作为阈值，扩大区别
                #self.c_rate = 0.98
                edge_idxs = list(np.argwhere(dis_matrix[0] < np.median(self.mean_dis_list)*self.c_r))
                #edge_idxs = list(np.argwhere(dis_matrix[0] < mean_dis))
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

                res = self.calculate_net_measures(self.G_list[ith_class], idx)

                #res =[]
                #res.append(aver_dis + self.mean_dis_list[ith_class]) #############
                #res.append(aver_dis) #############
                #res.append(mean_dis) #############

                new_measures.append(res)
                #self.draw_g()
                #测试插入后节点个数
                #print("after:", len(self.G_list[ith_class].nodes()), self.G_list[ith_class].nodes())
                #self.draw_g()
                self.G_list[ith_class].remove_node(idx)
                #print("after_remove:", len(self.G_list[ith_class].nodes()), self.G_list[ith_class].nodes())


            print(np.array(self.net_measures), np.array(new_measures))
            diff = paired_euclidean_distances(self.net_measures, new_measures)
            #print("diff:", diff)
            label = int(np.argmin(diff))
            print("p_label:", label)
            y_pred.append(label)   #返回数组中最小值得索引
            print("="*100)
            #print("neighbor:", neighbor)
            #self.classification(label, idx, neighbor)

        #G = nx.disjoint_union_all(self.G_list)
        #print("G_final:", len(G.nodes()), G.nodes())
        #self.draw_g()
        return np.array(y_pred)

    #不需要将新节点再插入到新网络中
    def classification(self, label, idx, neighbor):
        for ith_class in range(self.num_classes):
            if ith_class == label:
                self.G_list[ith_class].add_node(idx, value=None, typeNode="test")
                self.G_list[ith_class].add_edges_from(neighbor[ith_class])



    def check(self, x, y):
        y_hat = self.predict(x, y)  #predict函数中不能有y,此处只是为了验证而已
        print("origanl_y:", y)
        print("predict:", y_hat)
        acc = np.sum(y_hat == y) / len(y)
        #con_m = confusion_matrix(y, y_hat, labels=[0, 1, 2])
        con_m = self.draw_confusion_matrix(y, y_hat)
        #print("con_m:")
        #print(con_m)

        return acc, con_m

    def draw_confusion_matrix(self, y_true, y_pred):
        sns.set()
        f, ax = plt.subplots()
        C2 = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        #print("confusion_matrix:")  # 打印出来看看
        #print(C2)
        sns.heatmap(C2, annot=True, ax=ax)  # 画热力图

        ax.set_title('confusion matrix')  # 标题
        ax.set_xlabel('Predict')  # x轴
        ax.set_ylabel('True')  # y轴
        plt.show()

        return C2

    def calculate_net_measures(self, net, idx):


        #degree_assortativity = nx.degree_assortativity_coefficient(G=net)

        #average_clustering_coefficient = nx.average_clustering(G=net)
        #average_degree = np.mean([i[1] for i in nx.degree(net)])  # nx.degree 返回的是每个节点的度, 所以要获取再求平均
        #dimameter = nx.algorithms.distance_measures.diameter(net)
        #effic = nx.global_efficiency(G=net)
        #den = nx.density(G=net)
        #cc = nx.closeness_centrality(net)
        #katz  = nx.katz_centrality(net)
        #bet = nx.betweenness_centrality(net)
        cc = nx.rich_club_coefficient(net)
        if idx == []:
            sum_c = [cc.get(i, 0) for i in net.nodes()]
            new_cc = sum(sum_c) / len(net.nodes)
        else:
            new_cc = cc[idx]

        return [new_cc] #return [new_cc]


    def draw_g(self):
        color_map = {0: 'red', 1: 'green', 2: 'b', 3: 'y', 4: 'm', 5: 'c', 6: 'black',
                    7: 'grey', 8: 'y', 9: 'magenta'}
        plt.figure("Graph", figsize=(12, 12))
        G = nx.disjoint_union_all(self.G_list)
        color_list = []
        for idx, thisG in enumerate(self.G_list):
            color_list += [color_map[idx]] * len(thisG.nodes)

        pos = nx.spring_layout(G)

        nx.draw_networkx(G, pos, with_labels=False, node_size=40,
                         node_color=color_list, width=0.2, alpha=1)  #

        plt.show()

if __name__ == '__main__':

    start = time.time()

    def data_preprcess(x_train, x_test):
        min_max_scaler = preprocessing.MinMaxScaler().fit(x_train)
        x_train = min_max_scaler.transform(x_train)
        x_test = min_max_scaler.transform(x_test)
        return x_train, x_test

    #data = load_iris() #iris数据集
    #data = load_wine() #wine数据集
    #data = load_breast_cancer()
    #data = load_digits()
    #x = data['data']
    #y = data['target']

    #covid-19数据集
    x, y = get_data()    #covid-19数据集 label：0：normal， 1：P， 2：covid-19
    #print(x.shape, len(y))
    #df = pd.read_csv(r"C:\Users\Yan\Desktop\CovidData_n_min_4_1024.csv")
    #df = pd.read_csv(r"C:\Users\Yan\Desktop\CovidData_n_430.csv")
    #df = pd.read_csv(r"C:\Users\Yan\Desktop\Brazil study files\Dataset\fractal_demension.csv")

    """
    features = list(df.columns)
    features = features[: len(features) - 1]  # 去掉开头和结尾的两列数据
    x = df[features].values.astype(np.float32)
    y = np.array(df.target)
    """

    a = []
    con_matrix = []
    for i in range(10):
        print("frequency:", i)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
        x_train, x_test = data_preprcess(x_train, x_test) #数据归一化
        model = NetworkBaseModel()

        model.get_params(1, 0.95) #构建初始化网络的阈值和分类阈值
        model.fit(x_train, y_train)
        acc, con_m = model.check(x_test, y_test)

        print("acc and con_m:", acc, con_m)

        a.append(acc)
        con_matrix.append(con_m)

    #打印最终结果
    print("final:", a, con_matrix)
    mean_acc = np.mean(a)
    max = np.max(a)
    min = np.min(a)
    var = np.var(a)
    print("%f +- %f", (mean_acc, var))
    print(min, max)

    end = time.time()
    print("time:", end - start)

