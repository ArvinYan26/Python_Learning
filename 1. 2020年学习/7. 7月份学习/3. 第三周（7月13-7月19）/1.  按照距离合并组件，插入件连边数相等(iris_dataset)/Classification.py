from CaculateMeasures import calculate_measure
import numpy as np



def classification(self, insert_node_id, result):
    # 因为在构建离岸边的时候用的就是str()形式，所以需要此处需要转化为字符串
    adj = [n for n in G.neighbors(str(insert_node_id))]  # find the neighbors of the new node
    count0 = 0
    count1 = 0
    count2 = 0
    for n in adj:
        if n in G0.nodes():
            label = G._node[n]["label"]
            # print("label:", label)
            count0 += 1
        elif n in G1.nodes():
            label = G._node[n]["label"]
            # print("label:", label)
            count1 += 1
        elif n in G2.nodes():
            label = G._node[n]["label"]
            # print("label:", label)
            count2 += 1
    # print("edges_num:", count0, count1, count2)
    # 确认分类后我可以给节点添加标签，以防止新节点与其连接时不知道标签
    if count0 == len(adj):
        # print("classification_result:", 0)
        G.remove_node(str(insert_node_id))
        for n in adj:
            G.add_node(str(insert_node_id), typeNode='test', label=0)
            G.add_edge(str(insert_node_id), n)
        result.append(0)
    elif count1 == len(adj):
        # print("classification_result:", 1)
        G.remove_node(str(insert_node_id))
        for n in adj:
            G.add_node(str(insert_node_id), typeNode='test', label=1)
            G.add_edge(str(insert_node_id), n)
        result.append(1)
    elif count2 == len(adj):
        # print("classification_result:", 2)
        G.remove_node(str(insert_node_id))
        for n in adj:
            G.add_node(str(insert_node_id), typeNode='test', label=2)
            G.add_edge(str(insert_node_id), n)
        result.append(2)
    else:
        print("模糊分类情况：")
        # draw_graph(G)

        print(count0, count1, count2)
        dist_list = []
        if count0 >= 0 and count0 < len(adj):
            # delate the edges and node
            if str(insert_node_id) in G.nodes():
                G.remove_node(str(insert_node_id))
            # 找到类1中和adj中相同的节点，然后将节点添加到类1中
            node_list = G0.nodes()  # 这时候还是插入节点之前的G0
            neighbor = [x for x in node_list if x in adj]
            # 然后将节点添加到类0中
            for n in neighbor:
                G.add_node(str(insert_node_id), typeNode='test', label=0)
                G.add_edge(str(insert_node_id), n)
            get_subgraph()  # get the new sungraph to calclulate the measures
            measures0 = calculate_measure(G0)  # new subgraph G0 measures
            V1, V2 = np.array(net0_measure[len(net0_measure) - 1]), np.array(measures0)
            euclidean_dist0 = np.linalg.norm(V2 - V1)
            dist_list.append(euclidean_dist0)

        if count1 >= 0 and count1 < len(adj):
            if str(insert_node_id) in G.nodes():
                G.remove_node(str(insert_node_id))
            # 找到类1中和adj中相同的节点，
            node_list = G1.nodes()
            neighbor = [x for x in node_list if x in adj]
            # 然后将节点添加到类1中
            for n in neighbor:
                G.add_node(str(insert_node_id), typeNode='test', label=1)
                G.add_edge(str(insert_node_id), n)
            get_subgraph()
            measures1 = calculate_measure(G1)
            N1, N2 = np.array(net1_measure[len(net1_measure) - 1]), np.array(measures1)
            euclidean_dist1 = np.linalg.norm(N2 - N1)
            dist_list.append(euclidean_dist1)
        if count2 >= 0 and count2 < len(adj):
            if str(insert_node_id) in G.nodes():
                G.remove_node(str(insert_node_id))
            node_list = G2.nodes()
            neighbor = [x for x in node_list if x in adj]
            # 添加到2类网络中
            for n in neighbor:
                G.add_node(str(insert_node_id), typeNode='test', label=2)
                G.add_edge(str(insert_node_id), n)
            # draw_graph(G)
            get_subgraph()
            measures2 = calculate_measure(G2)
            M1, M2 = np.array(net2_measure[len(net2_measure) - 1]), np.array(measures2)
            # print("M1, M2:", M1, M2)
            euclidean_dist2 = np.linalg.norm(M2 - M1)
            dist_list.append(euclidean_dist2)
        # 确定模糊分类节点的分类标签后，需要将节点移除，然后重新插入到确定的分类标签处
        if str(insert_node_id) in G.nodes():
            G.remove_node(str(insert_node_id))
        # print(np.array(net0_measure), net1_measure, net2_measure,)
        print("dist_list:", dist_list)
        # get the classfication ruselt
        list = []
        for x in dist_list:
            if not x == 0:
                list.append(x)
        min_value = min(list)
        label = int(dist_list.index(min_value))
        print("classification_result:", label)
        result.append(label)

        if label == 0:
            node_list = G0.nodes()
            neighbor = [x for x in node_list if x in adj]
            # 然后将节点添加到类1中
            G.add_node(str(insert_node_id), typeNode='test', label=0)
            for n in neighbor:
                G.add_edge(str(insert_node_id), n)  # add edges

        if label == 1:
            node_list = G1.nodes()
            neighbor = [x for x in node_list if x in adj]
            # 然后将节点添加到类1中
            G.add_node(str(insert_node_id), typeNode='test', label=1)
            for n in neighbor:
                G.add_edge(str(insert_node_id), n)

        if label == 2:
            node_list = G2.nodes()
            neighbor = [x for x in node_list if x in adj]
            G.add_node(str(insert_node_id), typeNode='test', label=2)
            for n in neighbor:
                G.add_edge(str(insert_node_id), n)
        need_classification.append(str(insert_node_id))
    return result