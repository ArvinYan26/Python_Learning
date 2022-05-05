from sklearn.neighbors import NearestNeighbors
import numpy as np


def build_init_network(train_data, train_target, G, k, label):
    #DC = DataClassification(k, calss_num)
    for index, instance in enumerate(train_data):
        G.add_node(str(index), value=instance, typeNode="init_net", label=train_target[index])
    # print(nodes_list)

    # 切片范围必须是整型
    temp_nbrs = NearestNeighbors(k, metric='euclidean')
    temp_nbrs.fit(train_data)
    temp_nbrs.kneighbors(train_data)
    knn_distances, knn_indices = temp_nbrs.kneighbors(train_data)
    # print(" ")
    temp_radius = np.mean(knn_distances)   #平均值
    # print("temp_radius", radius)
    temp_nbrs.set_params(radius=temp_radius)
    radius_distances, radius_indices = temp_nbrs.radius_neighbors(train_data)
    # print(np.array(radius_indices))
    # 添加连边
    for idx, one_data in enumerate(train_data):  # 这个语句仅仅是获取索引indx，然后给他连边
        # print(knn_indices[idx], radius_indices[idx])
        if (len(radius_indices[idx])) > k:  # 判断用哪种方法连边构图,因为,radius应用于稠密区域，邻居大于k的话就是稠密
            # print("radius technique:")
            # print(idx, radius_indices[idx], knn_indices[idx])

            for indiceNode, indicesNode in enumerate(radius_indices):
                for tmpi, indice in enumerate(indicesNode):
                    if (str(indice) == str(indiceNode)):
                        continue
                    if (G.nodes()[str(indice)]["label"] == G.nodes()[str(indiceNode)][
                        "label"] or not label):
                        G.add_edge(str(indice), str(indiceNode), weight=radius_distances[indiceNode][tmpi])

        else:
            # print("KNN technique:")
            # print(idx, knn_indices[idx], radius_indices[idx])
            for indiceNode, indicesNode in enumerate(knn_indices):
                for tmpi, indice in enumerate(indicesNode):
                    if (str(indice) == str(indiceNode)):
                        continue
                    if (G.nodes()[str(indice)]["label"] == G.nodes()[str(indiceNode)][
                        "label"] or not label):
                        G.add_edge(str(indice), str(indiceNode), weight=knn_distances[indiceNode][tmpi])

        # do the next steps when there are thingle node (small components)
        # 单节点情况，和同类最近编号节点连边三个
        for idx, one_data in enumerate(train_data):
            count = 0
            new_id = str(idx)  # 因为neighbors迭代器用的节点编号的字符串，所以需要转化为字符串
            # print(G.nodes())
            adj = [n for n in G.neighbors(new_id)]  # find the neighbors of the new node
            if len(adj) == 0:
                count += 1
                # print("sing_node_num:", count)
                # print("idx:", idx)
                count = 0
                for i in range(len(G.nodes())):
                    if G._node[str(idx)]["label"] == G._node[str(i)]["label"]:
                        G.add_edges_from([(str(idx), str(i))])
                        count += 1
                    if count == k:
                        break
            # print("")

        return G, temp_nbrs, temp_radius

"""
#单节点情况，和同类最近编号节点连边三个
def check_single_node(train_data, train_target, G, k, label, nbrs):
    for idx, instance in enumerate(train_data):
        count = 0
        #new_id = str(idx)  # 因为neighbors迭代器用的节点编号的字符串，所以需要转化为字符串
        # print(G.nodes())
        adj = [n for n in G.neighbors(str(idx))]  # find the neighbors of the new node
        if len(adj) == 0:
            G.remove_node(str(idx))
            #count += 1
            # print("sing_node_num:", count)
            # print("idx:", idx)
            #count = 0

            G.add_node(str(idx), values=instance, typeNode='init_net', label=train_target[idx])
            # print(len(g.nodes()))

            knn_distances, knn_indices = nbrs.kneighbors([instance])
            radius_distances, radius_indices = nbrs.radius_neighbors([instance])
            # print(distances, indices)    #所以有有可能包含自身 ，到时候添加边要过滤掉
            # 添加到训练网络中
            if (len(radius_indices)) > k:  # 判断用哪种方法连边构图,因为,radius应用于稠密区域，邻居大于k的话就是稠密
                # print(radius_indices[idx])
                for indiceNode, indicesNode in enumerate(radius_indices):
                    for tmpi, indice in enumerate(indicesNode):
                        if (str(indice) == str(indiceNode)):
                            continue
                        if (G.nodes()[str(indice)]["label"] == G.nodes()[str(indiceNode)][
                            "label"] or not label):
                            G.add_edge(str(indice), str(idx), weight=radius_distances[indiceNode][tmpi])
            else:
                for indiceNode, indicesNode in enumerate(knn_indices):
                    for tmpi, indice in enumerate(indicesNode):
                        if (str(indice) == str(indiceNode)):
                            continue
                        if (G.nodes()[str(indice)]["label"] == G.nodes()[str(indiceNode)][
                            "label"] or not label):
                            G.add_edge(str(indice), str(idx), weight=knn_distances[indiceNode][tmpi])

    return G
"""

def node_insert(G, k, nbrs, instance, insert_node_id, label):
    """

    :param X_items: one node inserted
    :param nodeindex: the index of the inserted node
    :param Y_items: label of the inserted node
    :return:
    """
    # 插入新的节点构建连边
    G.add_node(str(insert_node_id), values=instance, typeNode='test', label=label)
    # print(len(g.nodes()))

    knn_distances, knn_indices = nbrs.kneighbors([instance])
    radius_distances, radius_indices = nbrs.radius_neighbors([instance])
    # print(distances, indices)    #所以有有可能包含自身 ，到时候添加边要过滤掉
    # 添加到训练网络中
    if (len(radius_indices)) > k:  # 判断用哪种方法连边构图,因为,radius应用于稠密区域，邻居大于k的话就是稠密
        # print(radius_indices[idx])
        for indiceNode, indicesNode in enumerate(radius_indices):
            for tmpi, indice in enumerate(indicesNode):
                if (str(indice) == str(indiceNode)):
                    continue
                G.add_edge(str(indice), str(insert_node_id), weight=radius_distances[indiceNode][tmpi])
    else:
        for indiceNode, indicesNode in enumerate(knn_indices):
            for tmpi, indice in enumerate(indicesNode):
                if (str(indice) == str(indiceNode)):
                    continue
                G.add_edge(str(indice), str(insert_node_id), weight=knn_distances[indiceNode][tmpi])
    return G