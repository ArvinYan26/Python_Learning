import matplotlib.pyplot as plt
import networkx as nx

def draw_graph(G):
    color_map = {0: 'red', 1: 'green', 2: 'yellow', 4: "k"}
    plt.figure("Graph", figsize=(9, 9))
    pos = nx.spring_layout(G)
    global color_list
    color_list = [color_map[G.nodes[node]["label"]] for node in G.nodes()]

    for index, (node, NodeType) in enumerate(G.nodes(data='NodeType')):
        if (NodeType == 'test'):
            color_list[index] = 'purple'
            # color_list = [self.color_map[G.nodes[node]['label']] for node in G.nodes()]
    nx.draw_networkx(G, pos, node_color=color_list, with_labels=True,
                     font_size=10, node_size=200)  # 节点默认大小为300
    plt.show()
