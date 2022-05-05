import networkx as nx


def calculate_measure(G):
    """
    :param net: 构建的网络g
    :param nodes: 每一类的网络节点
    :return:
    """
    """
    measures = []

    # 平均度，平均最短路径，平均聚类系数， 同配性， 传递性
    # 1.  平均度
    ave_deg = G.number_of_edges() * 2 / G.number_of_nodes()
    ave_deg = round(ave_deg, 3)
    # print("平均度为：%f" % ave_deg)
    measures.append(ave_deg)

    
    # 2.  平均最短路径长度(需要图是连通的)
    ave_shorest = nx.average_shortest_path_length(G)
    ave_shorest = round(ave_shorest, 3)
    #print("平均最短路径：", ave_shorest)
    #measures.append(ave_shorest)
    

    # 3.  平均聚类系数
    ave_cluster = nx.average_clustering(G)
    ave_cluster = round(ave_cluster, 3)
    # print("平均聚类系数：%f" % ave_cluster)
    measures.append(ave_cluster)

    # 4.  度同配系数 Compute degree assortativity of graph
    assortativity = nx.degree_assortativity_coefficient(G)
    assortativity = round(assortativity, 3)
    # print("同配性：%f" % assortativity)
    measures.append(assortativity)


    # 5.  传递性transitivity：计算图的传递性，g中所有可能三角形的分数。
    tran = nx.transitivity(G)
    tran = round(tran, 3)
    #print("三角形分数：%f" % tran)
    measures.append(tran)
    """
    measures = []
    """
    #紧密中心性
    closeness = nx.closeness_centrality(G)
    a = closeness[str(insert_node_id)]
    measures.append(a)

    eigenvector = nx.eigenvector_centrality(G)
    b = eigenvector[str(insert_node_id)]
    measures.append(b)
    """
    #density = nx.density(G)
    #measures.append(density)

    efficiency = nx.global_efficiency(G)
    measures.append(efficiency)

    return measures