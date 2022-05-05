import networkx as nx
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4, 5])
#G.add_edges_from([(1, 1), (2, 2), (1, 3), (3, 4), (1, 2)])
G.add_edges_from([(1, 3), (3, 4), (1, 2)])
print(G.edges)
#self_loop = list(nx.selfloop_edges(G))
#print(self_loop)
G.remove_edges_from(list(nx.selfloop_edges(G)))
print(G.edges)

"""
l = [1, 2, 3]
l.extend([1, 3, 4])
print(l)

dic = {1: 3, 2: 4, 3: 8}
print(dic[1])
"""
G = nx.path_graph(4)
nx.add_path(G, [10, 11, 12])
x = sorted(nx.connected_components(G), key=len, reverse=True)
print(x)