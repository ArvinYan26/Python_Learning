import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mlp
import numpy as np
#import igraph


def build_init_network():
    global G
    G = nx.Graph([(0, 1), (0, 2), (0, 3), (0, 7), (1, 2), (1, 3), (1, 9),
                  (2, 3), (2, 5), (3, 4), (3, 6), (3, 8)])

    draw_graph(G)
    draw_adj_matrix(G, 4)
    #G.add_edges_from([(10, 9), (8, 10)])
    #G.add_edge(5, 6)
    #draw_graph(G)

    return G

def draw_graph(G):
    plt.figure(figsize=(12, 12))
    nx.draw_networkx(G, node_color='r', node_size=1500, font_size=25) #大小默认是300
    plt.show()

def draw_adj_matrix(G, c_n):
    """

    :param adj_matrix:
    :param c_n: each length of data
    :return:
    """
    adj_matrix = np.array(nx.adjacency_matrix(G).todense())
    print("adj_matrix:", adj_matrix)
    m = np.zeros_like(adj_matrix) - 2
    size = adj_matrix.shape[0]
    m[:c_n, :c_n] = 0
    m[:c_n, c_n:] = 1
    m[c_n:, :c_n] = 1

    for i in range(size):
        m[i, i] = -1
    fig, ax = plt.subplots(figsize=(12, 12))

    colors = ['white', '#000000', '#6495ED', '#FF6A6A']
    # ax.matshow(m, cmap=plt.cm.Blues)
    cmap = mlp.colors.ListedColormap(colors)
    ax.matshow(m, cmap=cmap)

    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            v = adj_matrix[j, i]
            ax.text(i, j, int(v), va='center', ha='center', fontsize=20)
    plt.yticks(size=20)  # 设置纵坐标字体信息
    plt.xticks(size=20)
    #plt.xlabel("Threshold", fontsize=20)
    #plt.ylabel("Measure Value", fontsize=20)
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
katz = nx.katz_centrality(G)
print("katz:", katz)

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
#x = nx.rich_club_coefficient(G)
#print("rich clup:", x)

print(G.edges())
edges = []
for i in G.nodes:
    for j in G.nodes:
        if i > 7 and j > 7:
            edges.append((i, j))
            edges.append((j, i))
G.remove_edges_from(edges)
print(G.edges)