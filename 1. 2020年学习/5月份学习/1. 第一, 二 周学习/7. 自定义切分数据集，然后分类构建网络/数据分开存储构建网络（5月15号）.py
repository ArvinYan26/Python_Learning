import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from test import split_data   #test文件中的spli_data函数
from sklearn.metrics.pairwise import euclidean_distances
import time

class DataClassification(object):
    """iris数据集分类"""

    def __init__(self, k, num_class):
        self.k = k
        #self.g = []
        self.X_net, self.Y_net, self.X_items, self.Y_items = self.get_iris()
        self.data_len = len(self.X_net)  # 此程序是24
        self.num_class = num_class
        self.per_class_data_len = int(self.data_len / self.num_class)
        self.nbrs = []  #用来存储是哪个类别网络的nbrs
        self.radius = []  #用来存储是哪个类别的
        self.nodes_list = []
        self.edges_list = []
        self.color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'black'}
        self.net_measure = {}
        self.G0 = nx.Graph()
        self.G1 = nx.Graph()
        self.G2 = nx.Graph()
        self.build_X_train_network()



    def get_iris(self):
        """获取数据集"""
        iris = load_iris()
        iris_data = iris.data  #[:, 2:]
        iris_target = iris.target

        #存储切分后的数据，训练集和测试集
        X_train1 = []
        Y_train1 = []
        X_train2 = []
        Y_train2 = []
        train_data = []
        train_target = []

        #第一次划分，train_data, train_target （0.8比例，多数），  X_train1, Y_train1 （0.2，少数）
        train_data, train_target, X_train1, Y_train1  = split_data(iris_data, iris_target, X_train1, Y_train1, train_data, train_target)
        #print("训练集：")
        #print(np.array(X_train1), np.array(Y_train1))

        #第二次划分，X_net, Y_net（0.2中的0.8，多数训练集），   X_items, Y_items（0.2中的0.2部分，少数，用来测试插入节点）
        X_net, Y_net, X_items, Y_items = split_data(X_train1, Y_train1, X_train2, Y_train2, train_data, train_target)
        #X_net = self.data_preprocess(train_data)
        #X_items = self.data_preprocess(X_train1)
        """
        print("总的数据集:")
        print(len(np.array(X_train1)))
        print(np.array(X_train1), np.array(Y_train1))
        

        print("训练集：")
        print(len(X_net))
        print(X_net, Y_net)


        print("测试集：")
        print(np.array(X_items), np.array(Y_items))

        
        X_train, X_predict, Y_train, Y_predict = train_test_split(iris_data, iris_target, test_size=0.2)
        X_net, X_items, Y_net, Y_items = train_test_split(X_train, Y_train, test_size=0.2)
        #print(X_net)
        X_net = self.data_preprocess(X_net)
        #print(X_net)
        return X_net, Y_net, X_items, Y_items
        """
        return train_data, train_target, X_train1, Y_train1
        #return X_net, Y_net, X_items, Y_items  #

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

    def calculate_measure(self, G):
        """
        :param net: 构建的网络g
        :param nodes: 每一类的网络节点
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
        # 平均度，平均最短路径，平均聚类系数， 同配性， 传递性
        # 1.  平均度
        ave_deg = G.number_of_edges() * 2 / G.number_of_nodes()
        #print("平均度为：%f" % ave_deg)
        #measures.append(ave_deg)

        # 2.  平均最短路径长度(需要图是连通的)
        ave_shorest = nx.average_shortest_path_length(G)
        #print("平均最短路径：", ave_shorest)
        # measures.append(ave_shorest)

        # 3.  平均聚类系数
        ave_cluster = nx.average_clustering(G)
        #print("平均聚类系数：%f" % ave_cluster)
        #measures.append(ave_cluster)

        # 4.  度同配系数 Compute degree assortativity of graph
        assortativity = nx.degree_assortativity_coefficient(G)
        #print("同配性：%f" % assortativity)
        #measures.append(assortativity)

        # 5.  传递性transitivity：计算图的传递性，g中所有可能三角形的分数。
        tran = nx.transitivity(G)
        #print("三角形分数：%f" % tran)
        #measures.append(tran)

        return ave_deg, ave_shorest, ave_cluster, assortativity, tran

    def build_edges(self, G, i):

        print("类别：", i)
        current_data = self.X_net[self.per_class_data_len * i:self.per_class_data_len * (i + 1), :]
        print(current_data)
        for index, instance in enumerate(current_data):
            node_info = (index, {"value": list(instance), "class_num": i, "type": "train"})
            self.nodes_list.append(node_info)
        # print(self.nodes_list)

        # 切片范围必须是整型
        temp_nbrs = NearestNeighbors(self.k, metric='euclidean')
        temp_nbrs.fit(current_data)
        self.nbrs.append(temp_nbrs)  # 将每一类的nbrs都添加进列表
        knn_distances, knn_indices = self.KNN(temp_nbrs, current_data)
        #print(knn_distances, knn_indices)
        temp_radius = self.get_radius(knn_distances)
        self.radius.append(temp_radius)  # 将每一类的radius都添加进radius
        radius_distances, radius_indices = self.epsilon_radius(temp_nbrs, current_data, temp_radius)
        #print(radius_distances, radius_indices)
        # 添加连边
        for idx, one_data in enumerate(current_data):  # 这个语句仅仅是获取索引indx，然后给他连边
            #print(radius_indices[idx])
            if (len(radius_indices[idx])) > (self.k - 1):  # 判断用哪种方法连边构图,因为,radius应用于稠密区域，邻居大于k的话就是稠密
                #print(radius_indices[idx])
                for index, nbrs_indices in enumerate(radius_indices[idx]):
                    #print(index, nbrs_indices)
                    #for indices, eve_index in enumerate(nbrs_indices):
                        #print(indices, eve_index)
                    if idx == nbrs_indices:  # 如果是本身，就跳过，重新下一个循环
                        continue
                    edge = (idx, nbrs_indices, radius_distances[idx][index])
                    self.edges_list.append(edge)
            else:
                #print(idx, knn_indices[idx])
                for index, nbrs_indices in enumerate(knn_indices[idx]):
                    #print("信息")
                    #print(index, nbrs_indices)
                    #for indices, eve_index in enumerate(nbrs_indices):
                        #print(indices, eve_index)
                    if idx == nbrs_indices:  # 如果是本身，就跳过，重新下一个循环
                        continue
                    edge = (idx, nbrs_indices, knn_distances[idx][index])
                    self.edges_list.append(edge)

        G.add_weighted_edges_from(self.edges_list)
        # self.get_net_measures()  #
        #print(G.nodes())
        # color_list = [self.color_map[self.G.nodes[node]['class_num']] for node in self.G.nodes()]
        # plt.subplot(211)
        #print(self.X_net.shape)
        nx.draw_networkx(G, node_color=self.color_map[i], with_labels=True, node_size=300) #节点默认大小为300
        plt.title("X_net")
        plt.show()
        #print(self.G.nodes())
        #print("X_net节点数：", len(G.nodes()))
        #print("X_net边数：", len(G.edges()))

    def build_X_train_network(self):
        """
        分开构建网络
        :return:
        API reference
            klearn.neighbors.NearestNeighbors
                - https://scikit-learn.org/stable/modules/generated
        """
        for i in range(self.num_class): #按类别循环遍历每一类别的每一个数据
            if i == 0:
                self.build_edges(self.G0, i)
                self.net_measure[i] = self.calculate_measure(self.G0)
            if i == 1:
                self.build_edges(self.G1, i)
                self.net_measure[i] = self.calculate_measure(self.G1)
            if i == 2:
                self.build_edges(self.G2, i)
                self.net_measure[i] = self.calculate_measure(self.G2)

    def build_X_items_network(self, G, class_num, instance, node_name="new_node"):
        """
        添加每一个节点进是哪个类别网络
        :param class_num: 类别，针对不同的类用不同的nbrs
        :param instance:  节点值
        :param node_name: 新添加的节点name
        :return:
        """
        # insert_node_id = len(list(self.g.nodes()))
        # print(insert_node_id)

        print("添加的节点值：", instance)

        self.node_insert_num = len(G.nodes())
        print(self.node_insert_num)

        radius_distances, radius_indices = self.epsilon_radius(self.nbrs[class_num], [instance], self.radius[class_num])
        # print("radius:", radius_indices)
        distances, indices = self.KNN(self.nbrs[class_num], [instance])
        # print("knn:", indices)    #所以有有可能包含自身 ，到时候添加边要过滤掉

        # 添加到训练网络中
        # if 0 in distances:
        # return class_num

        edge_list = []
        G.add_node(self.node_insert_num, class_num=4, value=instance)  # 4：新颜色表示插入的新节点，用来分类
        if len(radius_indices) > (self.k - 1):
            # 其实此处只有一个实例输入进来，求出来的distance和indices也只是一维的
            for index, nbrs_indices in enumerate(radius_indices[0]):
                #for indices, eve_index, in enumerate(nbrs_indices):
                if self.node_insert_num == nbrs_indices:
                    continue
                edge = (nbrs_indices, self.node_insert_num, radius_distances[0][index])
                edge_list.append(edge)
        else:
            # 这里一定注意，是和那些邻居的节点索引链接，所以循环的是索引,否则会多出来很多边。
            #print(indices)
            for index, nbrs_indices in enumerate(indices[0]):
                #print(self.node_insert_num)
                #print(index, nbrs_indices)
                #for indices, eve_index in enumerate(nbrs_indices):
                    # print(indices, eve_index)
                if self.node_insert_num == nbrs_indices:
                    continue
                edge = (nbrs_indices, self.node_insert_num, distances[0][index])
                edge_list.append(edge)
        G.add_weighted_edges_from(edge_list)

        #print("添加边数：", len(edge_list))  #如果是0表示没有边添加进来
        #print(edge_list)
        # print("节点数：", len(self.g.nodes()))
        # print(self.g.edges())

        net_measure = self.calculate_measure(G)  #添加新节点后后的网络measures
        # 计算插入前插入后的相似度（此处用欧几里得距离）

        # 指标计算
        """
        print("指标值：")
        print("插入节点后的每一类指标值：", self.net_measure)
        print("class_num", class_num, self.net_measure[class_num])
        print("插入节点后的的指标：", net_measure)
        """
        v1, v2 = np.array(net_measure), np.array(self.net_measure[class_num])

        """
        print("v1后:", v1)
        print("v2:前", v2)
        """
        euclidean_distances = np.linalg.norm(v1 - v2) #插入后减去插入前
        #print("前后欧差：", euclidean_distances)

        #print("euclidean_distances:", euclidean_distances)

        return euclidean_distances, v1, v2

    def plot_insert_node(self, G, class_num):
        """

        :param G: 需要建立的图
        :param class_num: 类别名
        :return:
        """

        """
        #pos:参数里面的值必须是2维的，所以四维的用不了，此处不行，所以此参数不加
        pos = {}  #pos:用来存储要画的节点
        for i, v in enumerate(self.X_net):
            pos[i] = v
        print("pos:", pos)
        pos[node_name] = np.squeeze(instance)  #
        print("pos:", pos)
        """

        #color_list = [self.color_map.get(G.nodes[node]["class_num"]) for node in G.nodes()]
        plt.title("insert_node")
        nx.draw_networkx(G, with_labels=True, node_color=self.color_map[class_num], node_size=300)  # 节点大小默认值是300
        plt.show()

    def single_node_insert(self):
        for index, instance in enumerate(self.X_items):
            self.classicfication(instance)


        #画出最终的图
        self.plot_insert_node(self.G0, 0)
        print(len(self.G0.nodes()))

        self.plot_insert_node(self.G1, 1)
        print(len(self.G1.nodes()))

        self.plot_insert_node(self.G2, 2)
        print(len(self.G1.nodes()))



    def classicfication(self, instance, node_name="new_node"):
        """
        对新数据进行分类
        :param data: 新数据
        :return:
        """
        distances_list = [] #用于存放每类网络插入节点前和插入节点后的相似度
        #for index, instance in enumerate(data):
        for per_class in range(self.num_class):
            #print(per_class)
            #此处是将节点插入到每一类网络中去
            if per_class == 0:
                euclidean_distances, v1, v2 = self.build_X_items_network(self.G0, per_class, instance, node_name)

                print("v1后:", v1)
                print("v2:前", v2)
                print("前后欧差：", euclidean_distances)
                print(" ")

                distances_list.append(euclidean_distances)
                self.G0.remove_node(self.node_insert_num)
            if per_class == 1:
                euclidean_distances, v1, v2 = self.build_X_items_network(self.G1, per_class, instance, node_name)

                print("v1后:", v1)
                print("v2:前", v2)
                print("前后欧差：", euclidean_distances)
                print(" ")

                distances_list.append(euclidean_distances)
                self.G1.remove_node(self.node_insert_num)
            if per_class == 2:
                euclidean_distances, v1, v2 = self.build_X_items_network(self.G2, per_class, instance, node_name)

                print("v1后:", v1)
                print("v2:前", v2)
                print("前后欧差：", euclidean_distances)
                print(" ")

                distances_list.append(euclidean_distances)
                self.G2.remove_node(self.node_insert_num)

        print("欧差列表：", distances_list)
        class_num = distances_list.index(min(distances_list))
        print("类别名：", class_num)
        #print("result:", result)
        if class_num == 0:
            self.build_X_items_network(self.G0, class_num, instance, node_name)
            #self.plot_insert_node(self.G0, class_num)
            print("calssicfication:", class_num)
            print("*"*150)
            #print(self.G0.nodes())
            #print(len(self.G0.nodes()))

        if class_num == 1:
            self.build_X_items_network(self.G1, class_num, instance, node_name)
            #self.plot_insert_node(self.G1, class_num)
            print("calssicfication:", class_num)
            print("*"*150)
            #print(self.G1.nodes())
            #print(len(self.G1.nodes()))

        if class_num == 2:
            self.build_X_items_network(self.G2, class_num, instance, node_name)
            #self.plot_insert_node(self.G2, class_num)
            print("calssicfication:", class_num)
            print("*"*150)
            #print(self.G2.nodes())
            #print(len(self.G2.nodes()))


def main():
    dc = DataClassification(4, 3)
    #dc.classicfication([6.5, 3., 5.5, 1.8])
    dc.single_node_insert()


if __name__ == '__main__':
    main()