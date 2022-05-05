import networkx as nx

class GetSubgraph(object):
    def __init__(self, g, num):
        self.G = g
        self.class_num = num
        self.G0 = self.G1 = self.G2 = None
        self.target = None
    def get_subgraph(self):
        """得到各个类别网络中最大的组件"""
        num = nx.number_connected_components(self.G)
        Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)  #Gcc不能全局，因为会变化
        #print(Gcc)
        #for n in range(num):
        #前三个（0， 1， 2）不一定正好对应是哪个类别0,1,2， 也可能全是0类组件，所以不能只循环三个类别
        #用for循环的时候只会将最后三个集合的点构建为子图，我们要的是前几个


        for i in range(self.class_num): #class_num:保证了取到每个类别组件中最大的组件，然后用于合并小的组件
            #print("i：", i)
            G = self.G.subgraph(Gcc[i])
            #print(G.nodes())
            #for m in G.nodes():
                # print(X_train[int(m)])    #因为节点是字符串，所以必须转换位整型，才能知道数据
                # 分类阶段，数据没有标签，排除那些点影响，只循环遍历有标签的节点
                #后添加的节点都在最后，所以只需要看第一个节点label就行
                #if not G._node[0]["label"] is None:
            #list(G.nodes())[0]:G0中的节点列表的第一个元素
            for i in G.nodes():
                #print("i:", i)
                count = 0
                if not G._node[i]["label"] is None:
                    #print("label:", G._node[i]["label"])
                    count += 1
                    self.target = G._node[i]["label"] #子图节点集合中的第一个节点target
                    if count == 1:
                        break
            #print("self.target:", self.target)
            #print("target:", target)
            if self.target == 0:
                #if count0 > 3:  # 设置阈值，查找大的组件构建初试类网络，用以后边吞并小的组件
                #global G0
                if self.G0 is None:
                    self.G0 = G
                    #print("G0:", self.G0.nodes())
                elif self.G0 is not None:
                    continue
            elif self.target == 1:
                #if count1 > 3:
                #global G1
                if self.G1 is None:
                    self.G1 = G
                    #print("G1:", self.G1.nodes())
                elif self.G1 is not None:
                    continue
            elif self.target == 2:
                if self.G2 is None:
                    self.G2 = G
                    #print("G2:", self.G2.nodes())
                elif self.G2 is not None:
                    continue
            elif not self.G0 and self.G1 and self.G2 is None:
                break

        return self.G0, self.G1, self.G2