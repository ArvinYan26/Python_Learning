import networkx as nx
import numpy as np

def calculate_measure(G, x, y, insert_node_id):
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
    #密度
    density = nx.density(G)
    density = round(density, 3)
    measures.append(density)

    #全局效率
    efficiency = nx.global_efficiency(G)
    efficiency = round(efficiency, 3)
    measures.append(efficiency)

    #计算同类之间差别,作为measures
    data = [x[int(i)] for i in G.nodes()]
    if insert_node_id == []:
        #计算初始化网络的平均差别
        s = []
        for i in range(len(data)):
            for j in range(len(data)):
                if not i == j:
                    dis = np.linalg.norm(data[i] - data[j])
                    s.append(dis)
    else:
        # 计算新来的数据和所有过去数据的平均差别（着重强调新插入的数据）
        s = []
        for i in range(len(data)):
            if not i == insert_node_id:
                dis = np.linalg.norm(data[i] - x[insert_node_id]) #x[insert_node_id]:新插入的数据在原始数据集中的位置数据
                s.append(dis)


    #计算新来的数据和所有过去数据的平均相似度（着重强调新插入的数据）

    var = np.var(s)  #计算方差
    var = round(var, 3)
    measures.append(var)

    return measures