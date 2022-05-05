import matplotlib.pyplot as plt
import networkx as nx

class DataCalssification(object):

    def __init__(self):

        self.build_init_network()

    def build_init_network(self):
        self.G1 = nx.path_graph(10)
        """
        measures1 = self.calculate_measures(self.G1)
        rint("measures1:", measures1)  
        """

        self.G2 = nx.cycle_graph(10)
        plt.title("class1")
        self.plot_init_network(self.G2)
        """
        measures2 = self.calculate_measures(self.G2)
        print("measures2:", measures2)
        """

    def plot_init_network(self, G):
        plt.figure(figsize=(8, 7))
        nx.draw_networkx(G, with_labels=True, node_color='r', node_size=200)
        plt.show()

    def calculate_measures(self, G):

        measures = []
        # 平均度，平均最短路径，平均聚类系数， 同配性， 传递性
        # 1.  平均度
        ave_deg = G.number_of_edges() * 2 / G.number_of_nodes()
        ave_deg = round(ave_deg, 3)
        # print("平均度为：%f" % ave_deg)
        measures.append(ave_deg)

        # 3.  平均聚类系数
        ave_cluster = nx.average_clustering(G)
        ave_cluster = round(ave_cluster, 3)
        # print("平均聚类系数：%f" % ave_cluster)
        measures.append(ave_cluster)

        # 4.  度同配系数 Compute degree assortativity of graph
        assortativity = nx.degree_assortativity_coefficient(G)
        assortativity = round(assortativity, 3)
        # print("同配性：%f" % assortativity)
        measures.append(assortativity)

        return measures


    #def add_new_node(self):
