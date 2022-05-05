import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
import time

class DataClassification(object):
    """iris数据集分类"""
    def __init__(self, k, calss):
        self.k = k
        self.class_num = calss
        self.X_net, self.Y_net, self.X_items, self.Y_items = self.get_iris()
        self.nbrs, self.radius = self.build_X_train_network(label=True)
        self.g0, self.g1, self.g2 = self.get_subgraph(self.g)
        self.single_node_insert(self.X_items, self.Y_items)
        #self.color_map = {0: 'r', 1: 'y', 2: 'purple'}
        #self.radius = self.get_radius()
        #self.g = None  如果此处初始化None,那么其他任何函数最初调用它的时候都为空，而不是拿到了已经改变的self.g

    def get_iris(self):
        """获取数据集"""
        iris = load_iris()
        iris_data = iris.data
        iris_target = iris.target
        X_train = []

        X_train, X_predict, Y_train, Y_predict = train_test_split(iris_data, iris_target, test_size=0.2)
        X_net, X_items, Y_net, Y_items = train_test_split(X_train, Y_train, test_size=0.2)
        #print(X_net)
        X_net = self.data_preprocess(X_net)
        #print(X_net)
        return X_net, Y_net, X_items, Y_items

    def data_preprocess(self, data):
        """特征工程（归一化）"""
        #归一化
        scaler = preprocessing.MinMaxScaler().fit(data)
        data = scaler.transform(data)

        return data

    def KNN(self, nbrs, train_data):
        """
        KNN获取节点邻居和邻居索引
        :param nbrs:
        :param train_data:
        """
        distances, indices = nbrs.kneighbors(train_data)
        return distances, indices

    def get_radius(self, distances):

        return np.median(distances) #中位数

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

    def build_X_train_network(self, label):
        """
        :return:
            1. great network
            2. add nodes
            3. calculate distances and indices
            4. add edges
            5. plot graph
        API reference
            klearn.neighbors.NearestNeighbors
                - https://scikit-learn.org/stable/modules/generated
        """
        #print(self.X_net, self.X_items)
        self.g = nx.Graph()
        #print(len(self.X_net))
        #添加节点
        for index, instance in enumerate(self.X_net):
            self.g.add_node(str(index), values=instance, typeNode='net', label=self.Y_net[index])
            #print(index, instance)
        nbrs = NearestNeighbors(self.k, metric='euclidean')
        nbrs.fit(self.X_net)
        distances, indices = self.KNN(nbrs, self.X_net)
        #print(distances, indices)
        radius = self.get_radius(distances)
        #print(radius)
        radius_distances, radius_indices = self.epsilon_radius(nbrs, self.X_net, radius)
        #print(radius_distances, radius_indices)
        #添加连边
        #if self.radius / self.class_num > self.k
        if len(radius_indices[index]) > (self.k-1):  #判断用哪种方法连边构图,因为,radius应用于稠密区域，邻居大于k的话就是稠密
            for index, nbrs_indices in enumerate(radius_indices):
                for indices, eve_index in enumerate(nbrs_indices):
                    if index == eve_index:  #如果是本身，就跳过，重新下一个循环
                        continue
                    if self.g.nodes()[str(eve_index)]['label'] == self.g.nodes()[str(index)]['label']:
                        self.g.add_edge(str(eve_index), str(index), weight=radius_distances[index][indices])
        else:
            for index, nbrs_indices in enumerate(indices):
                for indices, eve_index in enumerate(nbrs_indices):
                    if index == eve_index: #如果是本身，就跳过，重新下一个循环
                        continue
                    if self.g.nodes()[str(eve_index)]['label'] == self.g.nodes()[str(index)]['label']:
                        self.g.add_edge(str(eve_index), str(index), weight=distances[index][indices])

        color_map = {0: 'r', 1: 'y', 2: 'purple'}
        color_list = [color_map[self.g.nodes[node]['label']] for node in self.g.nodes()]
        #plt.subplot(211)
        nx.draw_networkx(self.g, node_color=color_list, with_labels=False)
        plt.title("X_net")
        plt.show()
        #print(self.g.nodes())
        print("X_net节点数：", len(self.g.nodes()))
        print("X_net边数：", len(self.g.edges()))
        #number = nx.number_connected_components(self.g)
        #print(number)
        return nbrs, radius


    def single_node_insert(self, X_data, Y_label):
        """
        steps:
            1. 添加节点（for循环）
            2. 依次添加节点到三个子网络中，计算每个插入新节点后的子网络的5个指标，存储起来（列表）
            3. 计算每一个子网络插入前和插入后的网络指标相似度，存储起来
            4. 比较这三个相似度差值大小，将节点插入到差值最小的那个字网络中去
            5. 计算这个插入到子网络中去的节点在这个网络中最近的邻居，进行连边。

        :param X_items: one node inserted
        :param nodeindex: the index of the inserted node
        :param Y_items: label of the inserted node
        :return:
        """
        #添加节点
        for index, instance in enumerate(X_data):  #此时把节点插入到子图中，所以需要一个g接收子图
            print(index, instance)
            if (not len(Y_label) == 0):
                label = Y_label[index]
            #for g in [self.g0, self.g1, self.g2]:

            list_euclidean = []  #用于存放插入前插入后的相似度
            measures_before = np.array(self.calculate_measure(self.g0))
            #print("插入节点前方法：", measures_before)
            #list.append(measures_before)
            self.build_X_items_network(instance, self.g0,  label)
            measures_after = np.array(self.calculate_measure(self.g0))
            #print("插入节点后方法：", measures_after)
            measures_after = np.array(self.calculate_measure(self.g0))
            dist0 = np.linalg.norm(measures_after - measures_before)
            print("欧距0：", dist0)
            list_euclidean.append(dist0)

            measures_before = np.array(self.calculate_measure(self.g1))
            #print("插入节点前方法：", measures_before)
            #list.append(measures_before)
            self.build_X_items_network(instance, self.g1, label)
            measures_after = np.array(self.calculate_measure(self.g1))
            #print("插入节点后方法：", measures_after)
            measures_after = np.array(self.calculate_measure(self.g1))
            dist1 = np.linalg.norm(measures_after - measures_before)
            print("欧距1：", dist1)
            list_euclidean.append(dist1)

            measures_before = np.array(self.calculate_measure(self.g2))
            #print("插入节点前方法：", measures_before)
            #list.append(measures_before)
            self.build_X_items_network(instance, self.g2, label)
            measures_after = np.array(self.calculate_measure(self.g2))
            #print("插入节点后方法：", measures_after)
            measures_after = np.array(self.calculate_measure(self.g2))
            dist2 = np.linalg.norm(measures_after - measures_before)
            print("欧距2：", dist2)
            list_euclidean.append(dist2)
            print("组件欧差：", list_euclidean)


            if dist0 == min(list_euclidean):
                self.build_X_items_network(instance, self.g0, label)
                #self.plot_graph(self.g0)
            elif dist1 == min(list_euclidean):
                self.build_X_items_network(instance, self.g1, label)
                #self.plot_graph(self.g1)
            else:
                self.build_X_items_network(instance, self.g2, label)
                #self.plot_graph(self.g2)

            # euclidean = euclidean_distances(list)
            #print(list)
            #list = self.data_preprocess(list)
            #print(list)

        plt.subplot(234)
        self.plot_graph(self.g0)
        plt.subplot(235)
        self.plot_graph(self.g1)
        plt.subplot(236)
        self.plot_graph(self.g2)


        """
        nx.draw_networkx(self.g0, with_labels=False)
        plt.title("single_node_insert")
        plt.show()
        print("子图节点数：", len(self.g0.nodes()))
        print("子图边数：", len(self.g0.edges()))
        #self.get_subgraph(self.g0)
        """

    def build_X_items_network(self, instance, g, label=True):
        """
        将新的节点插入到相应的子网络（分类）中
        :return:
        """
        # insert_node_id = len(list(self.g.nodes()))
        # print(insert_node_id)

        insert_node_id = len(list(g.nodes()))
        print(insert_node_id)
        g.add_node(str(insert_node_id), values=instance, typeNode='test', label=label)
        #print(len(g.nodes()))

        radius_distances, radius_indices = self.nbrs.radius_neighbors([instance])
        distances, indices = self.nbrs.kneighbors([instance])
        #print(distances, indices)    #所以有有可能包含自身 ，到时候添加边要过滤掉
        print("邻居索引：", radius_indices)
        #添加到训练网络中
        if len(radius_indices[0]) > (self.k-1):
        #if self.radius/self.class_num > self.k:
            for index, nbrs_indices in enumerate(radius_indices):
                for indices, eve_index, in enumerate(nbrs_indices):
                    if index == eve_index:
                        continue
                    g.add_edge(str(eve_index), str(insert_node_id), weight=radius_distances[index][indices])
                    #print(len(g.edges()))
        else:
            #这里一定注意，是和那些邻居的节点索引链接，所以循环的是索引,否则会多出来很多边。
            for index, nbrs_indices in enumerate(indices):
                for indices, eve_index in enumerate(nbrs_indices):
                    if index == eve_index:
                        continue
                    g.add_edge(str(eve_index), str(insert_node_id), weight=distances[index][indices])

        #self.calculate_measure(self.g0)
        #color_map = {0: 'r', 1: 'y', 2: 'purple'}
        #color_list = [color_map[g.nodes[node]['label']] for node in self.g.nodes()]
        #plt.subplot(234)
        #number = nx.number_connected_components(self.g)
        #print(number)

    def plot_graph(self, g):
        """
        画图
        :param g: 需要画出来的图
        :return:
        """
        nx.draw_networkx(g, with_labels=False)
        plt.title("single_node_insert")
        plt.show()
        print("子图节点数：", len(g.nodes()))
        print("子图边数：", len(g.edges()))

    def get_subgraph(self, g):
        """
        先得到子图，然后在子图中进行插入接点
        :param g: network of the X_net data
        :return:
        """
        # 获得组件数量
        number = nx.number_connected_components(g)
        print("子图数:", number)
        print(" ")

        Gcc = sorted(nx.connected_components(g), key=len, reverse=True)
        g0 = self.g.subgraph(Gcc[0])
        g1 = self.g.subgraph(Gcc[1])
        g2 = self.g.subgraph(Gcc[2])

        g0 = nx.Graph(g0)
        print("g0节点数", len(g0.nodes()))
        #self.calculate_measure(g0)
        g1 = nx.Graph(g1)
        print("g1节点数", len(g1.nodes()))
        #self.calculate_measure(g1)
        g2 = nx.Graph(g2)
        print("g2节点数", len(g2.nodes()))
        print(" ")

        """
        self.calculate_measure(self.g0)
        plt.subplot(231)
        plt.title("subgraph_0")
        nx.draw_networkx(self.g0, with_labels=False)
        print("节点数：", len(self.g0.nodes()))
        print("连边：", len(self.g0.edges()))
        print(" ")

        self.g1 = self.g.subgraph(Gcc[1])
        self.calculate_measure(self.g1)
        plt.title("subgraph_1")
        plt.subplot(232)
        nx.draw_networkx(self.g1, with_labels=False)
        print("节点数：", len(self.g1.nodes()))
        print("边数：", len(self.g1.edges()))
        print(" ")

        self.g2 = self.g.subgraph(Gcc[2])
        self.calculate_measure(self.g2)
        plt.title("subgraph_2")
        plt.subplot(233)
        nx.draw_networkx(self.g2, with_labels=False)
        print("节点数：", len(self.g2.nodes()))
        print("边数：", len(self.g2.edges()))
        """
        return g0, g1, g2  #return 语句必须放在最后

    def calculate_measure(self, g):
        """
        :param g: X_net构建的网络图
        :return:
        """
        """
        des = nx.density(g)
        print("密度：%f" % des)
        
        # 度分布直方图
        # distribution = nx.degree_histogram(g)
        # print(distribution)
        measures.append(des)
        # 节点度
        deg = nx.degree(g)
        #print(deg)
        """
        measures = [] #平均度，平均最短路径，平均聚类系数， 同配性， 传递性
        #1.  平均度
        ave_deg = g.number_of_edges() * 2 / g.number_of_nodes()
        #print("平均度为：%f" % ave_deg)
        measures.append(ave_deg)

        #2.  平均最短路径长度(需要图是连通的)
        #ave_shorest = nx.average_shortest_path_length(g)
        #print("平均最短路径：", ave_shorest)
        #measures.append(ave_shorest)

        #3.  平均聚类系数
        ave_cluster = nx.average_clustering(g)
        #print("平均聚类系数：%f" % ave_cluster)
        measures.append(ave_cluster)

        #4.  同配性 Compute degree assortativity of graph
        assortativity = nx.degree_assortativity_coefficient(g)
        #print("同配性：%f" % assortativity)
        measures.append(assortativity)

        #5.  传递性transitivity：计算图的传递性，g中所有可能三角形的分数。
        tran = nx.transitivity(g)
        #print("三角形分数：%f" % tran)
        measures.append(tran)

        return measures

    def accurancy(self):
        """
        计算指标，正确率等， 用于评判
        :return:
        """
        pass


def main():
    DataClassification(5, 3)
    #如果再原函数中__init__方法中已经调用了函数，那么次数只需要给对象传参即可，其他函数无需再调用

    #dc.build_X_train_network(label=True)
    #time.sleep(1)
    #dc.single_node_insert(X_items, Y_items)
    #time.sleep(1)
    #dc.single_node_insert(X_predict, Y_predict)


if __name__ == '__main__':
    main()