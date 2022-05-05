import networkx as nx
import numpy as np
import BuildNetwork as BN

class GetSubgraph(object):
    def __init__(self, num_class, data, target, G, k, nbrs):
        """
        :param num_class: the num of the classes
        :param data: X_train or X_test
        """
        self.num_class = num_class
        self.data = data
        self.target = target
        self.G = G
        self.K = k
        self.nbrs = nbrs

    def get_subgraph(self):
        """得到各个类别网络中最大的组件"""
        num = nx.number_connected_components(self.G)
        Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)  #Gcc不能全局，因为会变化
        #print(Gcc)
        #for n in range(num):
        #前三个（0， 1， 2）不一定正好对应是哪个类别0,1,2， 也可能全是0类组件，所以不能支循环三个类别
        #用for循环的时候只会将最后三个集合的点构建为子图，我们要的是前几个
        self.G0 = self.G1 = self.G2 = None

        for i in range(len(Gcc)):
            G = self.G.subgraph(Gcc[i])
            #for m in G.nodes():
                # print(X_train[int(m)])    #因为节点是字符串，所以必须转换位整型，才能知道数据
                # 分类阶段，数据没有标签，排除那些点影响，只循环遍历有标签的节点
                #后添加的节点都在最后，所以只需要看第一个节点label就行
                #if not G._node[0]["label"] is None:
            #list(G.nodes())[0]:G0中的节点列表的第一个元素
            target = G._node[list(G.nodes())[0]]["label"] #子图节点集合中的第一个节点target
            if target == 0:
                #if count0 > 3:  # 设置阈值，查找大的组件构建初试类网络，用以后边吞并小的组件
                #global G0
                if self.G0 is None:
                    self.G0 = G
                    #print("G0:", self.G0.nodes())
                elif self.G0 is not None:
                    continue
            elif target == 1:
                #if count1 > 3:
                #global G1
                if self.G1 is None:
                    self.G1 = G
                    #print("G1:", self.G1.nodes())
                elif self.G1 is not None:
                    continue
            elif target == 2:
                if self.G2 is None:
                    self.G2 = G
                    #print("G2:", self.G2.nodes())
                elif self.G2 is not None:
                    continue
            elif not self.G0 and self.G1 and self.G2 is None:
                break

        return self.G0, self.G1, self.G2

    def merge_components(self):

        """
        合并单节点，小的组件。因为分类阶段不需要合并，所以这个函数需要单独建立
        :param G:
        :return:
        """
        #steps1：对于单节点，计算每一个单节点的最近的一个同类节点，然后连一条边，合并单节点
        for single_node in self.G.nodes():  # 单节点，因为neighbors迭代器用的节点编号的字符串，所以需要转化为字符串
            adj = [n for n in self.G.neighbors(single_node)]  # find the neighbors of the new node
            if len(adj) == 0: #说明是单节点
                all_dist = [] #所有节点对的节点编号和相对应的欧式距离
                for node_id in self.G.nodes(): #遍历每一个节点，找最近的节点
                    if single_node == node_id: #避免计算相同节点，形同节点距离是0，无法连边，没意义
                        continue
                    if self.G._node[single_node]["label"] == self.G._node[node_id]["label"]:
                        node_pair = [] #存放单节点和图中某一节点的节点对
                        dist = []  #计算单节点和当前图中某一个节点的相似度
                        node_pair.append(single_node)
                        node_pair.append(node_id)
                        dist.append(node_pair)
                        v2, v1 = np.array(self.data[int(node_id)]), np.array(self.data[int(single_node)])
                        d = np.linalg.norm(v2 - v1)
                        dist.append(d)
                        all_dist.append(dist)
                l = [a[1] for a in all_dist]  #将所有节点对的距离找出来
                index = l.index(min(l))    #找到最小节点对的索引
                #已经添加过节点了，所以不用再次添加节点，只需要连边就行了
                self.G.add_edge(all_dist[index][0][0], all_dist[index][0][1], weight=all_dist[index][1])

        #steps2: 小组件合并
        num = nx.number_connected_components(self.G)
        Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)
        #print(Gcc)
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
                            v2, v1 = np.array(self.data[int(m)]), np.array(self.data[int(n)])
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
                            """
                            print("one_all_dist:", one_all_dist)
                            print("min_value:", min_dist)
                            print("one_all_min:", one_all_min)
                            print("two_pars_min:", two_pars_min)
                            print("==" * 20)
                            """
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

        return self.G
