import networkx as nx
import matplotlib.pyplot as plt

l = [{'77', '82', '96', '32', '47', '57', '118', '29'}, {'86', '46', '95', '45', '73', '110', '98'},
     {'31', '23', '69', '49', '72'},  {'111', '36', '55'}]
#print("len:", len(l))
#set = l[0]
#print(set)
print(max(l))
G = nx.Graph()
l0 = l[0]

for n in range(len(l)):
    g = G.subgraph(l[n])
    print(list(g.nodes()))
    nx.draw_networkx(g)
    plt.show()

for n in l[0]:
    print("n:", n)