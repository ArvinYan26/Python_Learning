from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from normalization import data_preprocess
import networkx as nx
import numpy as np
import BuildNetwork as BN
from DrawGraph import draw_graph
from ReorganizeData import reorganize_data
from GetSubGraph import GetSubgraph
from CaculateMeasures import calculate_measure

class DataClassification(object):
    """
    classify dataset with the closeness_centrality
    """

    def __init__(self, k, num, Y_test):
        self.k = k
        self.class_num = num
        self.Y_test = Y_test
        self.G = None
        self.nbrs = {}
        self.radius = {}

    def fit(self, X_train, Y_train):
        """

        :param X_train: train data set
        :param Y_train:  train data traget
        :return:
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.G = nx.Graph()

    def get_params(self, deep=False):
        return {'k': self.k, 'num_class': self.class_num}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X_test, Y_test):
        self.predict(X_test, Y_test)

    def build_init_network(self):
        """
        build init network i.e. train network
        :return:
        """
        self.X_train, self.Y_train, class0, class1 , class2 = reorganize_data(self.X_train, self.Y_train)
        #print(new_data, len(new_data))
        #print(self.X_train, self.Y_train)
        #print(len(self.X_train), len(self.Y_train))

        self.G = nx.Graph()
        for index, instance in enumerate(self.X_train):
            #print(self.Y_train[index])
            self.G.add_node(index, value=instance, NodeType="train", label=self.Y_train[index])
        #draw_graph(self.G)
        for i in range(self.class_num):
            if i == 0:
                base_index = 0
                current_data =self.X_train[0:len(class0), :]
                #print(current_data, len(current_data))
                self.G, temp_nbrs, temp_radius = \
                    BN.build_init_network(current_data, base_index, self.G, self.k, label=True)
                self.nbrs["0"] = temp_nbrs
                self.radius["0"] = temp_radius

            if i == 1:
                base_index = len(class0)
                current_data = self.X_train[len(class0):(len(class0)+len(class1)), :]
                #print(current_data, len(current_data))
                self.G, temp_nbrs, temp_radius = \
                    BN.build_init_network(current_data, base_index, self.G, self.k, label=True)
                self.nbrs["1"] = temp_nbrs
                self.radius["1"] = temp_radius

            if i == 2:
                base_index = len(class0)+len(class1)
                current_data = self.X_train[(len(class0)+len(class1)):len(self.X_train), :]
                #print(current_data, len(current_data))
                self.G, temp_nbrs, temp_radius = \
                    BN.build_init_network(current_data, base_index, self.G, self.k, label=True)
                self.nbrs["2"] = temp_nbrs
                self.radius["2"] = temp_radius

        #print("self.nbrs:", self.nbrs, "self.radius:", self.radius)
        #print(self.G.nodes(), self.G.edges())
        #draw_graph(self.G)
        GSG = GetSubgraph(self.class_num, self.X_train, self.Y_train, self.G, self.k, self.nbrs)
        #steps1：合并小组件
        self.G = GSG.merge_components()
        self.G0, self.G1, self.G2 = GSG.get_subgraph()

        draw_graph(self.G)

        """
        #steps2：获得子图
        G0, G1, G2 = GSG.get_subgraph()
        self.G0 = G0.copy()
        self.G1 = G1.copy()
        self.G2 = G2.copy()
        draw_graph(self.G0)
        draw_graph(self.G1)
        draw_graph(self.G2)
        
        count0 = count1 = count2 = 0
        for n in self.G0.nodes:
            count0+= 1
            print(self.G0._node[n]["label"])
        print(count0, len(self.G0.nodes))
        for n in self.G1.nodes:
            count1 += 1
            print(self.G1._node[n]["label"])
        print(count1, len(self.G1.nodes))
        for n in self.G2.nodes:
            count2 += 1
            print(self.G2._node[n]["label"])
        print(count2, len(self.G2.nodes))
        """
    def predict(self, X_test, Y_test=[]):
        """

        :param X_test: test data set
        :param Y: test target
        :return:
        """
        self.X_test = X_test
        result = []
        # 添加节点
        for index, instance in enumerate(self.X_test):
            print("new_node_index:", index)
            #print("new_node_index:", index + len(list(self.G.nodes())))

            label = self.Y_test[index]
            print("new_node_label:", label)

            if (not len(Y_test) == 0):
                print("label:", self.Y_test[index])
            insert_node_id = len(list(self.G.nodes()))
            print("insert_node_id:", insert_node_id)

            all_varietion = []
            GSG = GetSubgraph(self.class_num, self.X_train, self.Y_train, self.G, self.k, self.nbrs)
            #插入前measures
            G0, G1, G2 = GSG.get_subgraph()
            #b_measures0 = calculate_measure(G0)
            #b_measures1 = calculate_measure(G1)
            #b_measures2 = calculate_measure(G2)

            G0 = G0.copy()
            G1 = G1.copy()
            G2 = G2.copy()
            """
            print(G0.nodes, len(G0.nodes))
            print(G1.nodes, len(G1.nodes))
            print(G2.nodes, len(G2.nodes))
            draw_graph(self.G0)
            draw_graph(self.G1)
            draw_graph(self.G2)
            """

            #插入后calculate the measures
            #添加节点进第一类
            measures = []
            base0_index = 0
            print(len(G0.nodes))
            G0 = BN.node_insert(G0, self.k, self.nbrs["0"], instance, base0_index, insert_node_id, label=4)
            adj0 = [n for n in G0.neighbors(insert_node_id)]
            print("adj0:", adj0)
            #draw_graph(G0)
            a_measure0 = calculate_measure(G0, insert_node_id)
            measures.append(a_measure0)
            #varietion0 = np.linalg.norm(np.array(a_measure0)-np.array(b_measures0))
            #all_varietion.append(varietion0)
            #print("b_measures0:", b_measures0, "a_measure0:", a_measure0)
            #if insert_node_id in G0.nodes():
                #G0.remove_node(insert_node_id)
            #draw_graph(G0)
            print("="*10)


            #添加节点到第二类
            base1_index = len(self.G0.nodes)
            print(len(G1.nodes))
            G1 = BN.node_insert(G1, self.k, self.nbrs["1"], instance, base1_index, insert_node_id, label=4)
            adj1 = [n for n in G1.neighbors(insert_node_id)]
            print("adj1:", adj1)
            #draw_graph(G1)
            a_measure1 = calculate_measure(G1, insert_node_id)
            measures.append(a_measure1)
            """
            varietion1 = np.linalg.norm(np.array(a_measure1)-np.array(b_measures1))
            all_varietion.append(varietion1)
            print("b_measures1:", b_measures1, "a_measure1:", a_measure1)
            if insert_node_id in G1.nodes():
                G1.remove_node(insert_node_id)
            draw_graph(G1)
            """
            print("=" * 10)

            #添加节点进第三类
            base2_index = len(self.G0.nodes) + len(self.G1.nodes)
            print(len(G2.nodes))
            G2 = BN.node_insert(G2, self.k, self.nbrs["2"], instance, base2_index, insert_node_id, label=4)
            adj2 = [n for n in G2.neighbors(insert_node_id)]
            print("adj2:", adj2)
            #draw_graph(G2)
            a_measure2 = calculate_measure(G2, insert_node_id)
            measures.append(a_measure2)
            """
            varietion2 = np.linalg.norm(np.array(a_measure2)-np.array(b_measures2))
            all_varietion.append(varietion2)
            print("b_measures2:", b_measures2, "a_measure2:", a_measure2)
            if insert_node_id in G2.nodes():
                G2.remove_node(insert_node_id)
            draw_graph(G2)
            """
            print("=" * 10)

            #print("all_varietion:", all_varietion)
            print("measures:", measures)
            predict_label = measures.index(max(measures))
            print("predict_label:", predict_label)

            if predict_label == 0:
                #adj = [n for n in self.G0.neighbors(insert_node_id)]
                print("0adj:", adj0)
                print(G0.nodes(), len(G0))
                self.G.add_node(insert_node_id, NodeType='test', label=0)
                for n in adj0:
                    self.G.add_edge(insert_node_id, n)
                result.append(0)
                #draw_graph(self.G)

            if predict_label == 1:
                #adj = [n for n in self.G1.neighbors(insert_node_id)]
                print("1adj0:", adj1)
                print(G1.nodes(), len(G1.nodes()))
                self.G.add_node(insert_node_id, NodeType='test', label=1)
                for n in adj1:
                    self.G.add_edge(insert_node_id, n)
                result.append(1)
                #draw_graph(self.G)

            if predict_label == 2:
                #adj = [n for n in self.G2.neighbors(insert_node_id)]
                print("2adj:", adj2)
                print(G2.nodes(), len(G2))
                self.G.add_node(insert_node_id, NodeType='test', label=2)
                for n in adj2:
                    self.G.add_edge(insert_node_id, n)
                result.append(2)
            #draw_graph(self.G)


            print("=="*50)
        print("self.Y_test:", self.Y_test)
        print("result_____:", result)
        draw_graph(self.G)
        GSG = GetSubgraph(self.class_num, self.X_train, self.Y_train, self.G, self.k, self.nbrs)
        # 插入前measures
        G0, G1, G2 = GSG.get_subgraph()

        """
        draw_graph(G0)
        draw_graph(G1)
        draw_graph(G2)
        """
        count = 0
        for i in range(len(self.Y_test)):
            if self.Y_test[i] == result[i]:
                count += 1
        print("正确个数：", count)
        accuracy = round(count / len(self.Y_test), 3)
        print("accuracy:", accuracy)




if __name__ == '__main__':
    data = load_iris()
    #data = load_wine()
    #data = load_breast_cancer()
    X = data.data
    y = data.target
    #X, y = make_moons(n_samples=200, noise=0)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_test = data_preprocess(X_train, X_test)
    DC = DataClassification(4, 3, Y_test)
    DC.fit(X_train, Y_train)
    DC.build_init_network()
    DC.predict(X_test)

