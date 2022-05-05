import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def build_init_network():
    global G
    G = nx.Graph([(0, 1), (0, 6), (1, 2), (1, 5), (5, 4), (2, 4), (2, 3), (4, 3), (3, 6), (7, 8), (7, 9), (8, 9)])
    draw_graph(G)
    G.add_edges_from([(10, 9), (8, 10)])
    G.add_edge(5, 6)
    draw_graph(G)

    return G

def draw_graph(G):
    nx.draw_networkx(G, node_color='r')
    plt.show()

def get_subgrph(G):
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    print(Gcc)
    print(list(Gcc[0]))
    print("type:", type(list(Gcc[0])))
    global G0, G1
    G0 = G.subgraph(Gcc[0])
    G0.graph["ClassName"] = 0
    G1 = G.subgraph(Gcc[1])
    G0.graph["ClassName"] = 1

    return G0, G1

G = build_init_network()
G0, G1 = get_subgrph(G)
G.add_nodes_from([11, 12, 13])

G.add_edges_from([(11, 10), (10, 12), (6, 13)])
G0, G1 = get_subgrph(G)
l = G0.nodes()
print("l:", l)
element = l[0]
print("element:", element)
print(list(G0.nodes())[2])
print(G0.nodes())

#for n in G0.nodes():


#G3 = G1.copy()


#global measures
"""
c = nx.communicability(G)
print("communicability:", c)
print(G.nodes())
x = c[3]
l = []
for values in x.values():
    l.append(values)
print("l:", l)
mean = np.mean(l)
print("communicability_mean:", mean)

efficiency = nx.global_efficiency(G)
print("efficiency:", efficiency)

betweenness = nx.betweenness_centrality(G)
print("betweenness:", betweenness)

"""
closeness = nx.closeness_centrality(G)
print("closeness:", closeness)

#group_closeness = nx.group_closeness_centrality(G)
#print("group_closeness:", group_closeness)
"""
#eigenvector:特征向量
eigenvector = nx.eigenvector_centrality(G)
print("eigenvector:", eigenvector)

measures = []
efficiency0 = nx.global_efficiency(G)
measures.append(efficiency0)

G.add_node(7)
G.add_edge(7, 4)
nx.draw_networkx(G)
plt.show()
efficiency1 = nx.global_efficiency(G)
measures.append(efficiency1)


G.add_node(8)
G.add_edge(8, 2)
nx.draw_networkx(G)
plt.show()
efficiency2 = nx.global_efficiency(G)
measures.append(efficiency2)
print(measures)
"""
betweenness = nx.betweenness_centrality(G)
print("betweenness:", betweenness)