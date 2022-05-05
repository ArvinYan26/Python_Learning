import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from ReorganizeData import reorganize_data
from GetSubgraph import GetSubgraph
from DrawGraph import draw_graph
class BuildNetwork(object):
    def __init__(self):
        self.G0 = self.G1 = self.G2 = None

    def build_init_network(self, x, y, mean_li, len_li, class_num):
        """

        :param x:train data
        :param y: train target
        :param mean_li: 每个类别相似度的平均值
        :param len_li: 每个类别数据长度
        :param class_num: 类别数
        :return: 构建的网络 G
        """
        self.G = nx.Graph()
        for index, instance in enumerate(x):
            self.G.add_node(index, value=instance, typeNode="init_net", label=y[index])
        #测试添加节点时是否成功
        #draw_graph(self.G)

        Matrix = np.round(euclidean_distances(x), decimals=3)  # 将矩阵中的元素保留相应小数点位数
        print("Matrix_len:", len(Matrix))
        InBegin = 0
        InEnd = len_li[0]

        for m in range(class_num):
            #print(InBegin, InEnd)
           #print(InBegin)
            for i in range(InBegin, InEnd):
                for j in range(InBegin, InEnd):
                    if i == j:
                        continue
                    # if adj_Matrix[i][j] <= mean_li[m]:
                    # print(mean_li[m])
                    # adj_Matrix[i][j] = adj_Matrix[j][i] =
                    # g.add_edge(i, j, weight=adj_Matrix[i][j])
                    if Matrix[i][j] > np.min(mean_li):
                        # print(mean_li[m])
                        Matrix[i][j] = Matrix[j][i] = 0
            # print(m)
            if m < class_num-1:
                InBegin += len_li[m]
                InEnd += len_li[m + 1]
        #print(Matrix)
        # 不同类之间不连边，
        OutBgin = 0
        OutEnd = len_li[0]  # 12
        for m1 in range(class_num - 1):  # 0-3, (0,1,2,)
            OutBgin1 = OutEnd
            OutEnd1 = OutEnd + len_li[m1+1]
            for m2 in range(class_num - m1 - 1):
                #print("+" * 100)
                #print(OutBgin, OutEnd)
                #print(OutBgin1, OutEnd1)
                for i in range(OutBgin, OutEnd):
                    for j in range(OutBgin1, OutEnd1):
                        Matrix[i][j] = Matrix[j][i] = 0
                if m1 < 1:
                    OutBgin1 += len_li[m1 + 1]
                    OutEnd1 += len_li[m1 + 1 + 1]
            if m1 < 2:
                OutBgin += len_li[m1]
                OutEnd += len_li[m1 + 1]
            #print("="*100)
            #print(OutBgin, OutEnd)
            #print(OutBgin1, OutEnd1)
        #print(Matrix)


        # 添加连边
        for i in range(len(Matrix)):
            for j in range(len(Matrix)):
                if not Matrix[i][j] == 0:
                    self.G.add_edge(i, j, weight=Matrix[i][j])

        #合并单节点和小组件++++++++++++++++++++++++++++++++++++++++++++++++++
        # steps1：对于单节点，计算每一个单节点的最近的一个同类节点，然后连一条边，合并单节点
        num = nx.number_connected_components(self.G)

        if num > class_num:
            for single_node in self.G.nodes():  # 单节点，因为neighbors迭代器用的节点编号的字符串，所以需要转化为字符串
                adj = [n for n in self.G.neighbors(single_node)]  # find the neighbors of the new node
                if len(adj) == 0:  # 说明是单节点
                    all_dist = []  # 所有节点对的节点编号和相对应的欧式距离
                    for node_id in self.G.nodes():  # 遍历每一个节点，找最近的节点
                        if single_node == node_id:  # 避免计算相同节点，形同节点距离是0，无法连边，没意义
                            continue
                        if self.G._node[single_node]["label"] == self.G._node[node_id]["label"]:
                            node_pair = []  # 存放单节点和图中某一节点的节点对
                            dist = []  # 计算单节点和当前图中某一个节点的相似度
                            node_pair.append(single_node)
                            node_pair.append(node_id)
                            dist.append(node_pair)
                            v2, v1 = np.array(x[int(node_id)]), np.array(x[int(single_node)])
                            d = np.linalg.norm(v2 - v1)
                            dist.append(d)
                            all_dist.append(dist)
                    l = [a[1] for a in all_dist]  # 将所有节点对的距离找出来
                    index = l.index(min(l))  # 找到最小节点对的索引
                    # 已经添加过节点了，所以不用再次添加节点，只需要连边就行了， 添加一条边就行
                    self.G.add_edge(all_dist[index][0][0], all_dist[index][0][1], weight=all_dist[index][1])

        """
        #steps2: 小组件合并（小的块）找两个组件中距离最小的两个节点，然后连边，按照距离合并组件
        num = nx.number_connected_components(self.G)
        Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)
        for i in range(len(Gcc)): #循环每一个子图中
            two_pars_min = [] #存储两部分中所有最近的两个节点标签和距离
            #print("i:", i)
            #print("Gcc[i]:", Gcc[i])
            for n in Gcc[i]:  #不影响使用，循环像影子图中的节点
                one_all_min = []
                count = 0  # 循环当前节点集合，然后跳出
                for x in range(i+1, len(Gcc)): #防止重复循环，所以从未计算过的子图节点开始计算

                    if self.G._node[list(Gcc[i])[0]]["label"] == self.G._node[list(Gcc[x])[0]]["label"]:
                        #print("x:", x)
                        #print("Gcc[x]:", Gcc[x])
                        one_all_dist = []
                        for m in Gcc[x]:
                            #print("n, m:", n, m)
                            node_pair = []  # 存放单节点和图中某一节点的节点对
                            dist = []  # 计算单节点和当前图中某一个节点的相似度
                            node_pair.append(n)
                            node_pair.append(m)
                            dist.append(node_pair)
                            print("x:", x)
                            v2, v1 = np.array(x[m]), np.array(x[n])
                            d = np.linalg.norm(v2 - v1)
                            dist.append(d)
                            one_all_dist.append(dist)
                        #print("=="*20)
                        if one_all_dist:  #如果是labela不一样，那么就没有距离，one_all_dist就会是空，
                            l = [m[1] for m in one_all_dist]  # 将所有节点对的距离找出来
                            index = l.index(min(l))
                            min_dist = one_all_dist[index]
                            one_all_min.append(min_dist)
                            two_pars_min.append(one_all_min[0]) #只将元素添加进来就行了所以one_all_min[0],取其元素
                            
                            print("one_all_dist:", one_all_dist)
                            print("min_value:", min_dist)
                            print("one_all_min:", one_all_min)
                            print("two_pars_min:", two_pars_min)
                            print("==" * 20)
                            
                        count += 1
                    if count == 1:
                        break

                if two_pars_min: #如果找不到同类，就下一个节点集合，所以需要判断，
                    s = [m[1] for m in two_pars_min]
                    #print("s:", s)
                    index = s.index(min(s))
                    min_dist = two_pars_min[index]
                    #print("min_dist:", min_dist)
                    self.G.add_edge(min_dist[0][0], min_dist[0][1], weight=min_dist[1])

        """
        print(num)
        num = nx.number_connected_components(self.G)
        if num > class_num:
            # 如果组件数大于类别数执行下面步骤,正常情况下，分类阶段不会用到，因为很明显的分3类
            GS = GetSubgraph(self.G, class_num)
            self.G0, self.G1, self.G2 = GS.get_subgraph() #得到每个类别中最大的组件，然后合并最小的组件
            #print(len(self.G.nodes))
            #print(len(self.G0.nodes) + len(self.G1.nodes) + len(self.G2.nodes))
            #print(num)
            Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)
            print("variance:", num - class_num)
            for m in range(num):
                #print("m:", m)
                G = self.G.subgraph(Gcc[m])
                for n in G.nodes():  #要循环小组件中的点，和自己同类的最大组件合并
                    if G._node[n]["label"] == 0:
                        count = 0
                        for a in self.G0.nodes():
                            if n == a:
                                continue
                            self.G.add_edge(n, a)
                            count += 1
                            if count == 1:  #尽可能让差别扩大，相似度高的类别内部本来就组建少，平均度高，影响不大，
                                            # 影响大的是同类差别大的
                                break
                    if G._node[n]["label"] == 1:
                        count = 0
                        for a in self.G1.nodes():
                            if n == a:
                                continue
                            self.G.add_edge(n, a)
                            count += 1
                            if count == 1:
                                break
                    if G._node[n]["label"] == 2:
                        count = 0
                        for a in self.G2.nodes():
                            if n == a:
                                continue
                            self.G.add_edge(n, a)
                            count += 1
                            if count == 1:
                                break

        return self.G
