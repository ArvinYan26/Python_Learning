#用for循环生成图

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

G = nx.Graph()
Matrix = np.array(
    [
        [0, 1, 0, 1, 1, 1, 0, 0],  # a
        [1, 0, 1, 0, 1, 0, 0, 0],  # b
        [0, 1, 0, 1, 0, 1, 0, 0],  # c
        [1, 0, 1, 0, 1, 0, 0, 0],  # d
        [1, 1, 0, 1, 0, 1, 0, 0],  # e
        [1, 0, 1, 0, 1, 0, 1, 1],  # f
        [0, 0, 0, 0, 0, 1, 0, 1],  # g
        [0, 0, 0, 0, 0, 1, 1, 0]  # h
    ]
)

"""
sum = 0
for i in range(len(Matrix)):
    for j in range(len(Matrix)):
        if Matrix[i][j] == 1:
            G.add_edge(i, j)
            sum += Matrix[i][j]
deg = nx.degree(G)
average_deg = sum / len(Matrix)
print(deg)
print(average_deg)
"""

degreesByNode=[]
for i in range(len(Matrix)):
    counterDegreeNode=0
    for j in range(len(Matrix)):
        if Matrix[i][j] == 1:
            counterDegreeNode += 1
            G.add_edge(i, j)
    degreesByNode.append(counterDegreeNode)
averageDegree = sum(degreesByNode) / len(Matrix)
print(G.degree())
print(averageDegree)

nx.draw(G, node_color='purple', node_size=300, with_labels=True)
plt.show()

"""
g = nx.Graph()
l = [(i, j) for i in range(25) for j in range(20)]
print(l)
g.add_edges_from(l)
nx.draw(g, node_color='b', node_size=300, with_labels=True)
plt.show()
"""