import networkx as nx
import matplotlib.pyplot as plt


pointList = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
linkList = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('E', 'F'), ('F', 'G')]

"""
def subgraph():
    g = nx.Graph()
    #转化为图结构
    g.add_nodes_from(pointList)
    g.add_edges_from(linkList)
    #draw graph
    plt.subplot(221)
    nx.draw(g, with_labels=True)

    color = ['y', 'g']
    subplot = [223, 224]
    #print subgraph
    graph = nx.connected_components(g)
    print(graph)
    for c in nx.connected_components(g):
        #得到不连通子图
        node_set = g.subgraph(c).nodes()
        #draw subgraph
        subgraph = g.subgraph(c)
        plt.subplot(subgraph[0])
        nx.draw_networkx(subgraph, with_labels=True, node_color=color[0])
        color.pop(0)
        subplot.pop(0)

    plt.show()

subgraph()

"""

pointList = ['A','B','C','D','E','F','G']
linkList = [('A','B'),('B','C'),('C','D'),('E','F'),('F','G')]

def subgraph():
    G = nx.Graph()
    # 转化为图结构
    for node in pointList:
        G.add_node(node)

    for link in linkList:
        G.add_edge(link[0], link[1])

   # 画图
    plt.subplot(211)
    nx.draw_networkx(G, with_labels=True)

    color =['y','g']
    subplot = [223,224]
    v = list(nx.connected_components(G))
    print(v)
    # 打印连通子图
    for c in v:
       # 得到不连通的子集
        nodeSet = G.subgraph(c).nodes()
       # 绘制子图
        subgraph = G.subgraph(c)
        plt.subplot(subplot[0])  # 第二整行
        nx.draw_networkx(subgraph, with_labels=True, node_color=color[0])
        color.pop(0)
        subplot.pop(0)

    plt.show()
subgraph()

"""
G = nx.path_graph(4)
nx.add_path(G, [10, 11, 12])
#l = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
#print(l)
plt.subplot(211)
nx.draw_networkx(G, with_labels=True, node_color='purple')

g = sorted(nx.connected_components(G), key=len, reverse=True)
plt.subplot(223)
g1 = G.subgraph(g[0])
deg = nx.degree(g1)
print("节点度：", deg)
nx.draw_networkx(g1)

plt.subplot(224)
g2 = G.subgraph(g[1])
nx.draw_networkx(g2)

plt.show()
"""