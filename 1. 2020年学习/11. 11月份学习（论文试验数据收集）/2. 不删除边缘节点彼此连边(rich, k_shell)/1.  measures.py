import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import igraph


def build_init_network():
    global G
    G = nx.Graph([(0, 1), (0, 2), (0, 3), (0,4), (0, 5), (1, 2), (1, 3), (1, 4),(1, 6), (2, 3), (2, 4),(2, 7), (3, 4), (3, 8), (8, 9)])
    #draw_graph(G)
    G.add_edges_from([(5, 11), (5, 10)])
    #draw_graph(G)

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

"""
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
"""

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

"""
closeness = nx.closeness_centrality(G)
print("closeness:", closeness)
katz = nx.katz_centrality(G)
print("katz:", katz)
"""

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


betweenness = nx.betweenness_centrality(G)
print("betweenness:", betweenness)
"""
draw_graph(G)
k_shell = nx.k_shell(G)
print(k_shell.nodes)
degree = nx.degree(k_shell)
print("degree:", degree)


"""
g = igraph.Graph(G)
draw_graph(g)
shell = g.shell_index(g)
print("shell:", shell)
"""

"""
draw_graph(G)
x = nx.rich_club_coefficient(G, normalized=False, Q=100)
print("rich clup:", x)
"""

G.add_edges_from([(4, 12), (3, 12), (2, 12), (1, 12)])
G.add_edges_from([(3, 13), (13, 14), (13, 15)])
G.add_edges_from([(4, 16), (3, 16), (16, 12), (16, 5), (16, 0), (16, 13)])

"""
draw_graph(G)
b = nx.rich_club_coefficient(G, normalized=False, Q=100)
print("rich clup:", b)
"""

draw_graph(G)
k_shell = nx.k_shell(G)
print(k_shell.nodes)
degree = nx.degree(k_shell)
print("degree:", degree)


