from CaculateMeasures import calculate_measure
import numpy as np
from GetSubGraph import GetSubgraph

class Classification(object):

    def __init__(self, G, num_class, X_train, nbrs, insert_node_id, net0, net1, net2):
        self.G = G
        self.num_class = num_class
        self.X_train = X_train
        self.nbrs = nbrs
        self.insert_node_id = insert_node_id
        self.net0_measure = net0
        self.net1_measure = net1
        self.net_measure = net2

    def classification(self, result, need_c):
        GSG = GetSubgraph(self.num_class, self.X_train, self.G)
        self.G0, self.G1, self.G2 = GSG.get_subgraph()
        # 因为在构建离岸边的时候用的就是str()形式，所以需要此处需要转化为字符串
        adj = [n for n in self.G.neighbors(str(self.insert_node_id))]  # find the neighbors of the new node
        print("adj:", adj)
        count0 = 0
        count1 = 0
        count2 = 0
        #self.get_subgraph()
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
        print("edges_num:", count0, count1, count2)
        #确认分类后我可以给节点添加标签，以防止新节点与其连接时不知道标签
        if count0 == len(adj):
            print("classification_result:", 0)
            if str(self.insert_node_id) in self.G.nodes():
                self.G.remove_node(str(self.insert_node_id))
            self.G.add_node(str(self.insert_node_id), typeNode='test', label=0)
            for n in adj:
                self.G.add_edge(str(self.insert_node_id), n)
            result.append(0)
        elif count1 == len(adj):
            print("classification_result:", 1)
            if str(self.insert_node_id) in self.G.nodes():
                self.G.remove_node(str(self.insert_node_id))
            self.G.add_node(str(self.insert_node_id), typeNode='test', label=1)
            for n in adj:
                self.G.add_edge(str(self.insert_node_id), n)
            result.append(1)
        elif count2 == len(adj):
            print("classification_result:", 2)
            if str(self.insert_node_id) in self.G.nodes():
                self.G.remove_node(str(self.insert_node_id))
            self.G.add_node(str(self.insert_node_id), typeNode='test', label=2)
            for n in adj:
                self.G.add_edge(str(self.insert_node_id), n)
            result.append(2)
        else:
            print("模糊分类情况：")
            #draw_graph(self.G)
            print(count0, count1, count2)
            dist_list = []
            if count0 == 0:
                dist_list.append(0)
            if count0 > 0 and count0 < len(adj):
                # delate the edges and node
                if str(self.insert_node_id) in self.G.nodes():
                    self.G.remove_node(str(self.insert_node_id))
                # 找到类1中和adj中相同的节点，然后将节点添加到类1中
                node_list = self.G0.nodes()  #这时候还是插入节点之前的G0
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类0中
                self.G.add_node(str(self.insert_node_id), typeNode='test', label=0)
                for n in neighbor:
                    self.G.add_edge(str(self.insert_node_id), n)
                GSG = GetSubgraph(self.num_class, self.X_train, self.G, self.nbrs)
                self.G0, self.G1, self.G2 = GSG.get_subgraph() #get the new sungraph to calclulate the measures
                measures0 = calculate_measure(self.G0)  # new subgraph self.G0 measures
                V1, V2 = np.array(self.net0_measure[len(self.net0_measure) - 1]), np.array(measures0)
                euclidean_dist0 = np.linalg.norm(V2 - V1)
                dist_list.append(euclidean_dist0)
            if count1 == 0:
                dist_list.append(0)
            if count1 > 0 and count1 < len(adj):
                if str(self.insert_node_id) in self.G.nodes():
                    self.G.remove_node(str(self.insert_node_id))
                # 找到类1中和adj中相同的节点，
                node_list = self.G1.nodes()
                neighbor = [x for x in node_list if x in adj]
                # 然后将节点添加到类1中
                self.G.add_node(str(self.insert_node_id), typeNode='test', label=1)
                for n in neighbor:
                    self.G.add_edge(str(self.insert_node_id), n)
                GSG = GetSubgraph(self.num_class, self.X_train, self.G, self.nbrs)
                self.G0, self.G1, self.G2 = GSG.get_subgraph()
                measures1 = calculate_measure(self.G1)
                N1, N2 = np.array(self.net1_measure[len(self.net1_measure) - 1]), np.array(measures1)
                euclidean_dist1 = np.linalg.norm(N2 - N1)
                dist_list.append(euclidean_dist1)

            if count2 == 0:
                dist_list.append(0)
            if count2 > 0 and count2 < len(adj):
                if str(self.insert_node_id) in self.G.nodes():
                    self.G.remove_node(str(self.insert_node_id))
                node_list = self.G2.nodes()
                neighbor = [x for x in node_list if x in adj]
                #添加到2类网络中
                self.G.add_node(str(self.insert_node_id), typeNode='test', label=2)
                for n in neighbor:
                    self.G.add_edge(str(self.insert_node_id), n)
                #self.draw_graph(self.G)
                GSG = GetSubgraph(self.num_class, self.X_train, self.G, self.nbrs)
                self.G0, self.G1, self.G2 = GSG.get_subgraph()
                measures2 = calculate_measure(self.G2)
                M1, M2 = np.array(self.net2_measure[len(self.net2_measure) - 1]), np.array(measures2)
                #print("M1, M2:", M1, M2)
                euclidean_dist2 = np.linalg.norm(M2 - M1)
                dist_list.append(euclidean_dist2)
            #确定模糊分类节点的分类标签后，需要将节点移除，然后重新插入到确定的分类标签处
            while str(self.insert_node_id) in self.G.nodes(): #防止之前的多次添加，while循环可以删除干净
                self.G.remove_node(str(self.insert_node_id))
            # print(np.array(self.net0_measure), self.net1_measure, self.net2_measure,)
            print("dist_list:", dist_list)
            # get the classfication ruselt
            list = []
            for x in dist_list:
                if not x == 0:
                    list.append(x)
            min_value = min(list)
            label = int(dist_list.index(min_value))
            print("classification_result:", label)
            result.append(label)

            if label == 0:
                node_list = self.G0.nodes()
                neighbor = [x for x in node_list if x in adj]
                print("neighbor0:", neighbor)
                # 然后将节点添加到类1中
                print("self.insert_node_id:", self.insert_node_id)
                for n in neighbor:
                    print(n)
                    self.G.add_node(str(self.insert_node_id), typeNode='test', label=0)
                    self.G.add_edge(str(self.insert_node_id), n)  # add edges
                adj = [n for n in self.G.neighbors(str(self.insert_node_id))]
                print("adj_new:", adj)
            if label == 1:
                node_list = self.G1.nodes()
                neighbor = [x for x in node_list if x in adj]
                print("neighbor1:", neighbor)
                # 然后将节点添加到类1中
                print("self.insert_node_id:", self.insert_node_id)
                for n in neighbor:
                    print(n)
                    self.G.add_node(str(self.insert_node_id), typeNode='test', label=1)
                    self.G.add_edge(str(self.insert_node_id), n)
                adj = [n for n in self.G.neighbors(str(self.insert_node_id))]
                print("adj_new:", adj)
            if label == 2:
                node_list = self.G2.nodes()
                neighbor = [x for x in node_list if x in adj]
                print("neighbor2:", neighbor)
                print("self.insert_node_id:", self.insert_node_id)

                for n in neighbor:
                    print(n)
                    self.G.add_node(str(self.insert_node_id), typeNode='test', label=2)
                    self.G.add_edge(str(self.insert_node_id), str(n))
                adj = [n for n in self.G.neighbors(str(self.insert_node_id))]
                print("adj:", adj)
            need_c.append(str(self.insert_node_id))

        return result, need_c
