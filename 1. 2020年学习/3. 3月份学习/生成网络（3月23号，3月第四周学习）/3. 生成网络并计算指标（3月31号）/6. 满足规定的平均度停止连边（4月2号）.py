import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

"""
#初始化参数
print("请输入网络节点总数：")
network_size = int(input())
print("请输入社区数：")
M = int(input())
in_nodes = int(network_size / M)
print("请输入社区内部连边概率：")
pin = float(input())
print("请输入社区外部连边概率：")
pout = float(input())

"""
network_size = 128
M = 8
in_nodes = int(network_size / M)
# 注意，如果此处in_nodes是4，那么第一个函数结束以后，它就变成了16，下一个函数再用时，就不再是4
pin = 0.7
pout = 0.01


# 生成全零矩阵
Matrix = np.zeros((network_size, network_size), dtype=int)

#初始化存储度和度分布的一维矩阵
degree = np.zeros((network_size), dtype=int)
degree_distribution = np.zeros((network_size), dtype=float)

def in_community():
    """社区内部创建边"""
    in_start = 0
    in_end = in_nodes
    for m in range(M):
        for i in range(in_start, in_end):
            for j in range(in_start, in_end):
                if i == j:
                    continue
                probability = np.random.random()
                if probability <= pin:
                    Matrix[i][j] = Matrix[j][i] = 1
        in_start += in_nodes
        in_end += in_nodes
    print(Matrix)


def generate_community():
    """社区之间创建边"""
    out_start = 0
    out_end = in_nodes
    # 最外层（第一层）循环遍历每一个社区
    for m in range(M - 1):
        out_start1 = out_end
        out_end1 = out_end + in_nodes
        # 第二层循环，遍历其他社区
        for m1 in range(M - m - 1):  # 如果不减去m,那么就会重复和每一个社区连边，而且包括它自己
            # 第三循环，遍历第一层循环指定的一个社内的所有节点
            for i in range(out_start, out_end):
                # 第四层循环，遍历其几个社区中指定的一个社区内部的所有节点与第一层循环控制的指定社区内的节点进行连边
                for j in range(out_start1, out_end1):
                    probability = np.random.random()
                    if probability <= pout:
                        Matrix[i][j] = Matrix[j][i] = 1
            out_start1 += in_nodes
            out_end1 += in_nodes
        out_start += in_nodes
        out_end += in_nodes
    print(Matrix)


def generate_graph():
    """生成图"""
    G = nx.Graph()
    count = 0
    for i in range(len(Matrix)):
        for j in range(len(Matrix)):
            if Matrix[i][j] == 1:
                G.add_edge(i, j)
                count += 1
                """
                for i in range(network_size):
                    for j in range(network_size):
                        degree[i] += Matrix[i][j]
                print(degree)
                sum_degree = 0
                for i in range(network_size):
                    sum_degree += degree[i]
                average_degree = float(sum_degree / network_size)
                print(average_degree)
                """

    print("构建的边数为：%d" % count)
    # 显示图
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    plt.show()

def deg():
    """计算节点度"""
    for i in range(network_size):
        for j in range(network_size):
            degree[i] += Matrix[i][j]
    print(degree)

def av_degree():
    """计算平均度"""
    sum_degree = 0
    for i in range(network_size):
        sum_degree += degree[i]
    average_degree = float(sum_degree / network_size)
    print(average_degree)

def calculate_degree_distribution():
    """计算度分布"""
    #1.统计度数为degree[i]的节点个数
    for i in range(network_size):
        degree_distribution[degree[i]] += 1
    print(degree_distribution)

    #2.计算度分布
    for i in range(network_size):
        degree_distribution[i] = degree_distribution[i] / network_size
    print(degree_distribution)


if __name__ == "__main__":
    in_community()
    generate_community()
    generate_graph()
    deg()
    average_degree = av_degree()
    calculate_degree_distribution()
