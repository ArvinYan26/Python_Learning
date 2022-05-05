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
        self.color_map = {0: 'red', 1: 'green', 2: 'yellow', 3: 'blue', -1: 'black'}
        #self.net_measure = {}
        self.build_X_train_network()



    def get_iris(self):
        """获取数据集"""
        iris = load_iris()
        iris_data = iris.data
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

        #第二次划分，train_data, train_target（0.2中的0.8，多数训练集），   X_train2, Y_train2（0.2中的0.2部分，少数，用来测试插入节点）
        X_net, Y_net, X_items, Y_items = split_data(X_train1, Y_train1, X_train2, Y_train2, train_data, train_target)

        """
        print("训练集：")
        print(len(X_net))
        print(X_net, Y_net)
        """
        print("测试集：")
        print(np.array(X_items), np.array(Y_items))

        """
        X_train, X_predict, Y_train, Y_predict = train_test_split(iris_data, iris_target, test_size=0.2)
        X_net, X_items, Y_net, Y_items = train_test_split(X_train, Y_train, test_size=0.2)
        #print(X_net)
        X_net = self.data_preprocess(X_net)
        #print(X_net)
        return X_net, Y_net, X_items, Y_items
        """
        return X_net, Y_net, X_items, Y_items  #

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

    def calculate_measure(self, net, nodes):
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
        #ave_deg = g.number_of_edges() * 2 / g.number_of_nodes()
        # print("平均度为：%f" % ave_deg)
        #measures.append(ave_deg)

        # 2.  平均最短路径长度(需要图是连通的)
        #ave_shorest = nx.average_shortest_path_length(g)
        # print("平均最短路径：", ave_shorest)
        # measures.append(ave_shorest)

        # 3.  平均聚类系数
        ave_cluster = nx.average_clustering(G=net, nodes=nodes)
        print("平均聚类系数：%f" % ave_cluster)
        #measures.append(ave_cluster)

        # 4.  度同配系数 Compute degree assortativity of graph
        assortativity = nx.degree_assortativity_coefficient(G=net, nodes=nodes)
        print("同配性：%f" % assortativity)
        #measures.append(assortativity)

        # 5.  传递性transitivity：计算图的传递性，g中所有可能三角形的分数。
        tran = nx.transitivity(G=net)
        print("三角形分数：%f" % tran)
        #measures.append(tran)

        return ave_cluster, assortativity, tran

    def get_net_measures(self):
        """
        按类别获取每一个类网络的measures
        :return:
        """
        self.net_measure = {}
        for per_class in range(self.num_class):
            node_list = range(per_class*self.per_class_data_len, (per_class+1)*self.per_class_data_len)
            self.net_measure[per_class] = self.calculate_measure(self.g, nodes=node_list)
        print("ever_measure:", self.net_measure)

    def build_X_train_network(self):
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
        #print(len(self.X_net))
        #添加节点
        self.g = nx.Graph()
        label_num = 0
        for index, instance in enumerate(self.X_net):
            #nodeinfo:可以可无
            node_info =(index, {"value": list(instance), "class_num": label_num, "type": "train"})
            if (index + 1) % self.per_class_data_len == 0:
                label_num += 1
            self.nodes_list.append(node_info)
        self.g.add_nodes_from(self.nodes_list)

        for i in range(self.num_class): #按类别循环遍历每一类别的每一个数据
            base_index = self.per_class_data_len*i
            #print(type(self.per_class_data_len))
            #print(type(i))
            #print(type(self.per_class_data_len*i))
            #切片范围必须是整型
            current_data = self.X_net[self.per_class_data_len*i:self.per_class_data_len*(i+1), :]
            temp_nbrs = NearestNeighbors(self.k, metric='euclidean')
            temp_nbrs.fit(current_data)
            self.nbrs.append(temp_nbrs)   #将每一类的nbrs都添加进列表
            knn_distances, knn_indices = self.KNN(temp_nbrs, current_data)
            #print(distances, indices)
            temp_radius = self.get_radius(knn_distances)
            self.radius.append(temp_radius)   #将每一类的radius都添加进radius
            radius_distances, radius_indices = self.epsilon_radius(temp_nbrs, current_data, temp_radius)
            #print(radius_distances, radius_indices)
            #添加连边
            for idx, one_data in enumerate(current_data): #这个语句仅仅是获取索引indx，然后给他连边
                #print(idx, one_data)
                if len(radius_indices[idx]) > (self.k-1):  #判断用哪种方法连边构图,因为,radius应用于稠密区域，邻居大于k的话就是稠密
                    for index, nbrs_indices in enumerate(radius_indices):
                        for indices, eve_index in enumerate(nbrs_indices):
                            if index == eve_index:  #如果是本身，就跳过，重新下一个循环
                                continue
                            edge = (idx + base_index, eve_index + base_index, radius_distances[index][indices])
                            self.edges_list.append(edge)
                else:
                    for index, nbrs_indices in enumerate(knn_indices):
                        #print("信息")
                        #print(index, nbrs_indices)
                        for indices, eve_index in enumerate(nbrs_indices):
                            #print(indices, eve_index)
                            if index == eve_index: #如果是本身，就跳过，重新下一个循环
                                continue
                            edge = (idx + base_index, eve_index + base_index, knn_distances[index][indices])
                            self.edges_list.append(edge)

        self.g.add_weighted_edges_from(self.edges_list)

        self.get_net_measures()  #

        print(self.g.nodes())
        color_list = [self.color_map[self.g.nodes[node]['class_num']] for node in self.g.nodes()]
        #plt.subplot(211)
        print(self.X_net.shape)
        nx.draw_networkx(self.g, node_color=color_list, with_labels=True, node_size=100)
        plt.title("X_net")
        plt.show()
        #print(self.g.nodes())
        print("X_net节点数：", len(self.g.nodes()))
        print("X_net边数：", len(self.g.edges()))

    def build_X_items_network(self, class_num, instance, node_name="new_node"):
        """
        添加每一个节点进是哪个类别网络
        :param class_num: 类别，label
        :param instance:  节点值
        :param node_name: 新添加的节点name
        :return:
        """
        # insert_node_id = len(list(self.g.nodes()))
        # print(insert_node_id)

        print("添加的节点值：", instance)
        base_index = self.per_class_data_len*class_num

        radius_distances, radius_indices = self.epsilon_radius(self.nbrs[class_num], [instance], self.radius[class_num])
        #print("radius:", radius_indices)
        distances, indices = self.KNN(self.nbrs[class_num], [instance])
        #print("knn:", indices)    #所以有有可能包含自身 ，到时候添加边要过滤掉

        #添加到训练网络中
        #if 0 in distances:
            #return class_num

        edge_list = []
        self.g.add_node(node_name, class_num=-1, value=instance) #-1：新颜色表示插入的新节点，用来分类
        if len(radius_indices) > (self.k-1):
        #其实此处只有一个实例输入进来，求出来的distance和indices也只是一维的
            for index, nbrs_indices in enumerate(radius_indices):
                for indices, eve_index, in enumerate(nbrs_indices):
                    if index == eve_index:
                        continue
                    edge = (base_index+eve_index, node_name, radius_distances[0][indices])
                    edge_list.append(edge)
        else:
            #这里一定注意，是和那些邻居的节点索引链接，所以循环的是索引,否则会多出来很多边。
            for index, nbrs_indices in enumerate(indices):
                #print(index, nbrs_indices)
                for indices, eve_index in enumerate(nbrs_indices):
                    #print(indices, eve_index)
                    if index == eve_index:
                        continue
                    edge = (base_index+eve_index, node_name, distances[0][indices])
                    edge_list.append(edge)
        self.g.add_weighted_edges_from(edge_list)
        #print("添加边数：", len(edge_list))  #如果是0表示没有边添加进来
        #print(edge_list)
        #print("节点数：", len(self.g.nodes()))
        #print(self.g.edges())

        current_class_nodes = list(range(base_index, base_index+self.per_class_data_len)).append("new_node")
        net_measure = self.calculate_measure(net=self.g, nodes=current_class_nodes)
        #计算插入前插入后的相似度（此处用欧几里得距离）

        #指标计算
        """
        print("指标值：")
        print(self.net_measure)
        print(self.net_measure[class_num])
        print(net_measure)
        """
        v1, v2 = np.array(self.net_measure[class_num]), np.array(net_measure)
        euclidean_distances = np.linalg.norm(v1-v2)
        print("euclidean_distances:", euclidean_distances)

        return euclidean_distances, net_measure

    def plot_insert_node(self, instance, node_name="new_node"):
        """

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

        color_list = [self.color_map[self.g.nodes[node]["class_num"]] for node in self.g.nodes()]
        plt.title("insert_node")
        nx.draw_networkx(self.g,  with_labels=True, node_color=color_list, node_size=100)  # 节点大小默认值是300
        plt.show()

    def get_single_node(self):
        """
        遍历测试数据集，获得每一个节点
        :param data:  测试集数据
        :return:
        """
        for index, instance in enumerate(self.X_items):
            #print(instance)
            self.classicfication(instance)

    def classicfication(self, instance, node_name="new_node"):
        """
        对新数据进行分类
        :param data: 新数据
        :return:
        """
        distances_list = [] #用于存放没类网络插入节点前和插入节点后的相似度
        #for index, instance in enumerate(data):
        for per_class in range(self.num_class):
            #print(per_class)
            #此处是将节点插入到每一类网络中去
            euclidean_distances, measure = self.build_X_items_network(per_class, instance, node_name)
            distances_list.append(euclidean_distances)
            self.g.remove_node(node_name)
        #print("欧差列表：", distances_list)
        result = distances_list.index(min(distances_list))
        #print("result:", result)
        self.build_X_items_network(result, instance, node_name)
        self.plot_insert_node(instance)
        print("calssicfication:", result)
        print("*"*30)
        print(self.g.nodes())

        print(len(self.g.nodes()))



def main():
    dc = DataClassification(5, 3)
    #如果再原函数中__init__方法中已经调用了函数，那么次数只需要给对象传参即可，其他函数无需再调用

    dc.classicfication([5.4, 3.4, 1.7, 0.2])

    #dc.get_single_node()

if __name__ == '__main__':
    main()
