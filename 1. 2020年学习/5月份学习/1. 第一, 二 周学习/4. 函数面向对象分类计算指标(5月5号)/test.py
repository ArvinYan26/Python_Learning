import networkx as nx
G = nx.random_graphs.barabasi_albert_graph(1000,3)   #生成一个n=1000，m=3的BA无标度网络


print(G.degree(0))                                  #返回某个节点的度
print(G.degree())                                     #返回所有节点的度
print(nx.degree_histogram(G))

import random
color_map = {0: 'red', 1: 'green', 2: 'yellow', 3: 'blue', -1: 'black'}
s = color_map.get(random.randint(-1, 4), 0)
print(s)