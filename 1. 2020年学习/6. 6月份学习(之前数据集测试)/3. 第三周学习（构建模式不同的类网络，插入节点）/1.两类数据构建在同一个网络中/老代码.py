import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from test import split_data   #test文件中的spli_data函数
from make_simple_dataset import generate_dataset

import time

class DataClassification(object):
    """iris数据集分类"""

    def __init__(self, k, num_class):
        self.k = k
        #self.g = []
        self.X_net, self.Y_net, self.X_items, self.Y_items, self.X_test, self.Y_test = self.get_iris()
        self.data_len = len(self.X_net)  # 此程序是24
        self.num_class = num_class
        self.per_class_data_len = int(self.data_len / self.num_class)
        self.plot_data(self.X_net)  #直接执行此函数
        self.nbrs = []  #用来存储是哪个类别网络的nbrs
        self.radius = []  #用来存储是哪个类别的
        self.weight_alpha = [0.1, 0.4, 0.4]  #measures权重
        self.color_map = {0: 'red', 1: 'green'}
        self.net0_measure = []   #存储每一类别插入Y_items中的数据时的指标
        self.impact0 = []   #storage the impact of the first class
        self.net1_measure = []
        self.impact1 = []
        self.G = nx.Graph()

        self.predict_target = []

        #初始化运行程序，一开始就运行
        self.build_X_train_network()
        self.add_X_items_data()
        self.single_node_insert()
        self.accuracy()

    def get_data(self):
        """获取数据集"""
        """
        iris = load_iris()
        iris_data = iris.data  #[:, 2:]
        iris_target = iris.target
        """
        data, label = generate_dataset()

        #存储切分后的数据，训练集,和测试集
        X_train1 = []
        Y_train1 = []
        train_data = []
        train_target = []
        X_train2 = []
        Y_train2 = []
        X_train3 = []
        Ytrain3 = []

        #第一次划分，train_data, train_target （0.8比例，多数），  X_train1, Y_train1 （0.2，少数）
        train_data, train_target, X_test, Y_test  = split_data(data, label, X_train1, Y_train1, train_data, train_target)

        #第二次划分，X_net 0.8, X_items 0.2
        X_net, Y_net, X_items, Y_items = split_data(train_data, train_target, X_train2, Y_train2, X_train3, Ytrain3)
        #print("训练集：")
        #print(np.array(X_train1), np.array(Y_train1))



        print("总的数据集:")
        print(data, label)
        print("X_traing")
        print(train_data, train_target)
        print("X_net:")
        print(X_net, Y_net)
        print("X_items：")
        print(np.array(X_items), np.array(Y_items))
        print("X_test:")
        print(np.array(X_test), np.array(Y_test))


        return X_net, Y_net, X_items, Y_items, X_test, Y_test


    def plot_data(self, data):
        """画出数据"""
        node_style = ["ro", "go"]
        for i in range(self.num_class):
            plt.plot(data[self.per_class_data_len*i:self.per_class_data_len*(i+1), 0],
                     data[self.per_class_data_len * i:self.per_class_data_len * (i + 1), 1],
                     node_style[i],
                     label=node_style[i])
            plt.legend()
        plt.show()



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
        measures = []

        # 平均度，平均最短路径，平均聚类系数， 同配性， 传递性
        # 1.  平均度
        ave_deg = G.number_of_edges() * 2 / G.number_of_nodes()
        ave_deg = round(ave_deg, 3)
        #print("平均度为：%f" % ave_deg)
        measures.append(ave_deg)

        """
        # 2.  平均最短路径长度(需要图是连通的)
        ave_shorest = nx.average_shortest_path_length(G)
        ave_shorest = round(ave_shorest, 3)
        #print("平均最短路径：", ave_shorest)
        #measures.append(ave_shorest)
        """

        # 3.  平均聚类系数
        ave_cluster = nx.average_clustering(G)
        ave_cluster = round(ave_cluster, 3)
        #print("平均聚类系数：%f" % ave_cluster)
        measures.append(ave_cluster)

        # 4.  度同配系数 Compute degree assortativity of graph
        assortativity = nx.degree_assortativity_coefficient(G)
        assortativity = round(assortativity, 3)
        #print("同配性：%f" % assortativity)
        measures.append(assortativity)

        """
        # 5.  传递性transitivity：计算图的传递性，g中所有可能三角形的分数。
        tran = nx.transitivity(G)
        tran = round(tran, 3)
        #print("三角形分数：%f" % tran)
        measures.append(tran)
        """


        return measures

    def build_edges(self, G, i):
        edges_list = []
        nodes_list = []
        # print("类别：", i)
        #current_data = self.X_net[self.per_class_data_len * i:self.per_class_data_len * (i + 1), :]
        # print(current_data)
        for index, instance in enumerate(self.X_net):
            node_info = (index, {"value": list(instance), "label": self.Y_net[index], "type": "train"})
            nodes_list.append(node_info)
        # print(self.nodes_list)

        # 切片范围必须是整型
        temp_nbrs = NearestNeighbors(self.k, metric='euclidean')
        temp_nbrs.fit(self.X_net)
        self.nbrs.append(temp_nbrs)  # 将每一类的nbrs都添加进列表， 这个
        knn_distances, knn_indices = self.KNN(temp_nbrs, self.X_net)
        # print(knn_distances, knn_indices)
        temp_radius = self.get_radius(knn_distances)

        self.radius.append(temp_radius)  # 将每一类的radius都添加进radius
        print("temp_radius", self.radius)
        radius_distances, radius_indices = self.epsilon_radius(temp_nbrs, self.X_net, temp_radius)
        # print(radius_distances, radius_indices)
        # 添加连边
        for idx, one_data in enumerate(self.X_net):  # 这个语句仅仅是获取索引indx，然后给他连边
            # print(radius_indices[idx])
            if (len(radius_indices[idx])) > self.k:  # 判断用哪种方法连边构图,因为,radius应用于稠密区域，邻居大于k的话就是稠密
                # print(radius_indices[idx])
                for index, nbrs_indices in enumerate(radius_indices[idx]):
                    # print(index, nbrs_indices)
                    # for indices, eve_index in enumerate(nbrs_indices):
                    # print(indices, eve_index)
                    if idx == nbrs_indices:  # 如果是本身，就跳过，重新下一个循环
                        continue
                    edge = (idx, nbrs_indices, radius_distances[idx][index])
                    edges_list.append(edge)
            else:
                # print(idx, knn_indices[idx])
                for index, nbrs_indices in enumerate(knn_indices[idx]):
                    # print("信息")
                    # print(index, nbrs_indices)
                    # for indices, eve_index in enumerate(nbrs_indices):
                    # print(indices, eve_index)
                    if idx == nbrs_indices:  # 如果是本身，就跳过，重新下一个循环
                        continue
                    edge = (idx, nbrs_indices, knn_distances[idx][index])
                    edges_list.append(edge)

        G.add_weighted_edges_from(edges_list)
        nx.draw_networkx(G, node_color=self.color_map[i], with_labels=True, node_size=300)  # 节点默认大小为300
        plt.title("X_net")
        plt.show()
        # print(self.G.nodes())
        # print("X_net节点数：", len(G.nodes()))
        # print("X_net边数：", len(G.edges()))

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
                net0_measures = self.calculate_measure(self.G0)
                self.net0_measure.append(net0_measures)
            if i == 1:
                self.build_edges(self.G1, i)
                net1_measures = self.calculate_measure(self.G0)
                self.net1_measure.append(net1_measures)
            """
            if i == 2:
                self.build_edges(self.G2, i)
                self.net_measure[i] = self.calculate_measure(self.G2)
            """

    def build_X_items_network(self, G, class_num, instance):
        """
        添加每一个节点进是哪个类别网络
        :param class_num: 类别，针对不同的类用不同的nbrs
        :param instance:  节点值
        :param node_name: 新添加的节点name
        :return:
        """
        # insert_node_id = len(list(self.g.nodes()))
        # print(insert_node_id)
        print(G.edges())
        print("添加的节点值：", instance)

        self.node_insert_num = len(G.nodes())  #新插入
        print("self.node_insert_num", self.node_insert_num)

        radius_distances, radius_indices = self.epsilon_radius(self.nbrs[class_num], [instance], self.radius[class_num])
        # print("radius:", radius_indices)
        distances, indices = self.KNN(self.nbrs[class_num], [instance])
        # print("knn:", indices)    #所以有有可能包含自身 ，到时候添加边要过滤掉

        # 添加到训练网络中
        # if 0 in distances:
        # return class_num

        edge_list = []
        G.add_node(self.node_insert_num, class_num=4, value=instance)  # 4：新颜色表示插入的新节点，用来分类
        if len(radius_indices) > self.k:
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
        #print(class_num)
        #color_list = self.color_map.get(class_num)
        #print(color_list)
        #color_list = [self.color_map.get(G.nodes[node]["class_num"]) for node in G.nodes()]
        # nx.draw_networkx(G, with_labels=True, node_color=color_list, node_size=300)

        plt.title("insert_node")
        nx.draw_networkx(G, with_labels=True, node_color=self.color_map[class_num], node_size=300)  # 节点大小默认值是300
        plt.show()

    def add_X_items_data(self):
        """add the data of X_items one by one"""
        for index, instance in enumerate(self.X_items):
            #print(instance, self.Y_items[index])
            if self.Y_items[index] == 0:
                print(self.Y_items[index], instance)
                #for i in range(5000):
                #下面语句可以考虑在同一个图中构建三类网络
                self.build_X_items_network(self.G0, int(self.Y_items[index]), instance)
                print(self.net0_measure)
                net0_measure = self.calculate_measure(self.G0)
                self.net0_measure.append(net0_measure)
                #variation = np.array(net0_measure)-np.array(self.net0_measure[])
            elif self.Y_items[index] == 1:
                print(self.Y_items[index], instance)
                self.build_X_items_network(self.G1, int(self.Y_items[index]), instance)
                net1_measure = self.calculate_measure(self.G1)
                self.net1_measure.append(net1_measure)
        print("net0_measure:")
        print(np.array(self.net0_measure))
        print("net1_measure:")
        print(np.array(self.net1_measure))
        print(len(self.net0_measure))
        for i in range(len(self.net0_measure)-1):
            #0类网络变化
            M1 = np.array(self.net0_measure[i+1])
            M0 = np.array(self.net0_measure[i])
            variation0 = (M1 - M0) / M0
            self.impact0.append(variation0)

            #1类网络变化
            N1 = np.array(self.net1_measure[i+1])
            N0 = np.array(self.net1_measure[i])
            variation1 = (N1 - N0) / N0
            self.impact1.append(variation1)

        print("self.impact0:")
        print(np.array(self.impact0))
        print("self.impact1:")
        print(np.array(self.impact1))
        print("="*100)


    def single_node_insert(self):
        """
        add the X_test data noe by one
        :return:
        """
        for index, instance in enumerate(self.X_test):
            self.classicfication(instance)

        # 画出最终的图
        self.plot_insert_node(self.G0, 0)
        print(len(self.G0.nodes()))
        print(self.G0.edges())

        self.plot_insert_node(self.G1, 1)
        print(len(self.G1.nodes()))
        print(self.G1.edges())

        """
        self.plot_insert_node(self.G2, 2)
        print(len(self.G2.nodes()))
        print(self.G2.edges())
        """

    def classicfication(self, instance):
        """
        对新数据的每一个数据进行分类
        :return:
        """
        distances_list = []  # 用于存放每类网络插入节点前和插入节点后的相似度

        im0 = np.array(self.impact0)
        mean_impact0 = np.mean(im0, axis=0)   #原始的，不再变化
        print("mean_impact0:", mean_impact0)



        #将节点插入到第0类网络中##############################################
        print("G0 length:", len(self.G0.nodes()))
        print(self.net0_measure)
        self.build_X_items_network(self.G0, 0, instance)
        net0_measure = self.calculate_measure(self.G0)
        #self.net0_measure.append(net0_measure)
        print("add_new_bode:")
        print("插入后：", len(self.G0.nodes()))
        print(self.G0.edges())

        #计算插入节点后的measures和之前的measures的variation0(差异：variation)
        i = len(self.net0_measure)
        #for i in range(len(self.net0_measure)):
        print(self.net0_measure[i-1], self.net0_measure[i-2])
        M1 = np.array(net0_measure)
        M0 = np.array(self.net0_measure[i-1])
        variation0 = (M1 - M0) / M0
        print("variation0:", variation0)
        v1, v2 = np.array(mean_impact0), np.array( variation0)
        distance0 = np.linalg.norm(v1 - v2)
        print("distance0:", distance0)
        distances_list.append(distance0)


        #print("前后欧差：", euclidean_distances)

        """
        print("impact0 length:", len(self.impact0))
        #calculate the euclidean_distance between now and past
        euclidean_distance = []
        for i in range(len(self.impact0)):
            v1, v2 = np.array(variation0), np.array(self.impact0[i])
            print(v1, v2)


            distance = np.linalg.norm(v1 - v2)
            euclidean_distance.append(distance)
        """
        #print(euclidean_distance)
        #mean_0 = np.mean(euclidean_distance)
        #print("mean_0:", mean_0)
        #distances_list.append(mean_0)
       #distances_list.append(euclidean_distances)
        self.G0.remove_node(self.node_insert_num)
        print("移除后：", len(self.G0.nodes()))
        print(" ")

        #将节点添加到第1类网络中##################################################
        im1 = np.array(self.impact1)
        mean_impact1 = np.mean(im1, axis=0)  # 原始的，不再变化
        print("mean_impact1:", mean_impact1)

        print("G1 length:", len(self.G1.nodes()))
        print(self.net1_measure)
        self.build_X_items_network(self.G1, 1, instance)
        net1_measure = self.calculate_measure(self.G1)
        self.net1_measure.append(net1_measure)
        print("add_new_bode:")
        print(self.net1_measure)
        print("插入后：", len(self.G1.nodes()))
        print(self.G1.edges())

        # 计算插入节点后的measures和之前的measures的variation0(差异：variation)
        i = len(self.net0_measure)
        print(self.net0_measure[i - 1], self.net0_measure[i - 2])
        N1 = np.array(net1_measure)
        N0 = np.array(self.net1_measure[i - 1])
        variation1 = (N1 - N0) / N0
        print("variation1:", variation1)
        v1, v2 = np.array(mean_impact1), np.array(variation1)
        distance1 = np.linalg.norm(v1 - v2)
        print("distance0:", distance1)
        distances_list.append(distance1)

        """
        euclidean_distance = []
        for i in range(len(self.impact1)):
            v1, v2 = np.array(variation1), np.array(self.impact1[i])
            print(v1, v2)
            distance = np.linalg.norm(v1 - v2)
            #计算现在与过去影响的相似度（）
            euclidean_distance.append(distance)
        print(euclidean_distance)
        mean_1 = np.mean(euclidean_distance)
        print("mean_1:", mean_1)
        """

        #mean_num = np.mean(va)
        #distances_list.append(euclidean_distances)
        self.G1.remove_node(self.node_insert_num)
        print(len(self.G1.nodes()))
        print(" ")
        """
        if per_class == 2:
            euclidean_distances, v1, v2 = self.build_X_items_network(self.G2, per_class, instance)
            print(len(self.G2.nodes()))
            print(self.G2.edges())
            print("v1后:", v1)
            print("v2:前", v2)
            print("前后欧差：", euclidean_distances)


                distances_list.append(euclidean_distances)
                self.G2.remove_node(self.node_insert_num)
                print(len(self.G2.nodes()))
                print(" ")
            """
        print("欧差列表：", distances_list)
        class_num = distances_list.index(min(distances_list))
        print("class_name:", class_num)
        print("$"*150)

        if class_num == 0:
            print("分类前：", len(self.G0.nodes()))
            self.build_X_items_network(self.G0, class_num, instance)
            print("分类后：", len(self.G0.nodes()))
            # 插入节点后，网络measures变化，所以需要更新原来字典里的measures
            net0_measure = self.calculate_measure(self.G0)
            self.net0_measure.append(net0_measure)
            print(np.array(self.net0_measure))
            self.impact0.append(variation0)
            print("self.impact0:", np.array(self.impact0))

            print("calssicfication:", class_num)
            self.predict_target.append(class_num)
            print("*"*150)
            #print(self.G0.nodes())
            #print(len(self.G0.nodes()))

        if class_num == 1:
            print("分类前：", len(self.G1.nodes()))
            self.build_X_items_network(self.G1, class_num, instance)
            print("分类后：", len(self.G1.nodes()))
            # 插入节点后，网络measures变化，所以需要更新原来字典里的measures
            net0_measure = self.calculate_measure(self.G1)
            self.net1_measure.append(net1_measure)
            print(np.array(self.net1_measure))
            self.impact1.append(variation1)
            print("self.impact1:", np.array(self.impact1))

            print("calssicfication:", class_num)
            self.predict_target.append(class_num)
            print("*"*150)
            #print(self.G1.nodes())
            #print(len(self.G1.nodes()))

        """
        if class_num == 2:
            print("分类前：", len(self.G2.nodes()))
            self.build_X_items_network(self.G2, class_num, instance)
            print("分类后：", len(self.G2.nodes()))
            #插入节点后，网络measures变化，所以需要更新原来字典里的measures
            #self.net_measure[class_num] = self.calculate_measure(self.G2)
            print(self.net_measure)
            print("calssicfication:", class_num)
            self.predict_target.append(class_num)
            print("*"*150)
            #print(self.G2.nodes())
            #print(len(self.G2.nodes()))
        """

    def accuracy(self):
        """
        calculate  the accuracy of classification
        :return:
        """
        label = list(map(int, self.Y_items))  # 廖雪峰，高阶函数内容
        print("original_label:", label)
        print("predict_label :", self.predict_target)

        count = 0
        for i in range(len(self.Y_items)):
            if self.Y_items[i] == self.predict_target[i]:
                count += 1
        print(count)
        accuracy = round(count / len(self.Y_items), 3)
        print("accuracy:", accuracy)



if __name__ == "__main__":
    DataClassification(3, 2)
