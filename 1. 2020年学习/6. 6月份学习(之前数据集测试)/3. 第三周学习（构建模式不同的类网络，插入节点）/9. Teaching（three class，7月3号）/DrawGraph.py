import matplotlib.pyplot as plt
import networkx as nx

def draw_graph(G):
    color_map = {1: 'red', 2: 'green', 3: 'yellow'}
    plt.figure("Graph", figsize=(9, 9))
    pos = nx.spring_layout(G)
    color_list = [color_map[G.nodes[node]['label']] for node in G.nodes()]

    for index, (node, typeNode) in enumerate(G.nodes(data='typeNode')):
        if (typeNode == 'test'):
            color_list[index] = 'purple'
            # color_list = [self.color_map[G.nodes[node]['label']] for node in G.nodes()]
    nx.draw_networkx(G, pos, node_color=color_list, with_labels=True, node_size=200)  # 节点默认大小为300
    plt.show()
    # print("node_num：", len(G.nodes()))
    # print("edges_num：", len(G.edges()))
    # print(" ")