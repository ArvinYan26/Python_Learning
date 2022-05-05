import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

#设置网络大小和连边概率
print("请输入网络节点数：")
network_size = int(input())
print("请输入连边概率：")
probability_of_edge = float(input())

start_time = time.time() #开始时间
#生成全零矩阵
Matrix = np.zeros((network_size, network_size), dtype=int)

def generate_adjacency_matrix():
    """生成邻接矩阵，即生成网络。同时统计边的个数"""
    count = 0
    for i in range(network_size):
        for j in range(i+1, network_size):
            probability = np.random.random()
            if probability <= probability_of_edge:
                Matrix[i][j] = Matrix[j][i] = 1
                count += 1
    print(Matrix)
    print("构建的网络的边数为：%d"% count)

def generate_graph():
    """画出图，并且显示出来"""
    G = nx.Graph()
    for i in range(len(Matrix)):
        for j in range(len(Matrix)):
            if Matrix[i][j] == 1:
                G.add_edge(i, j)
    print(Matrix)
    nx.draw(G)
    plt.show()

def write_adjacency_matrix_to_file():
    """将生成的邻接矩阵写进文档内"""
    L = []
    f = open("AdjacencyMatrix.txt", "w+")
    for i in range(network_size):
        t = Matrix[i]
        L.append(t)
        for j in range(network_size):
            s = str(t[j])
            f.write(s)
            f.write("  ")
        f.write("\n")
    f.close()




def calculate_degree_distribution():
    """将度分布写进文档内"""
    degree = np.zeros((network_size), dtype=int)
    sum_degree = 0
    degree_distribution = np.zeros((network_size), dtype=float)

    #计算节点度
    for i in range(network_size):
        for j in range(network_size):
            degree[i] = degree[i] + Matrix[i][j]
    print(degree)

    #计算平均度
    for i in range(network_size):
        sum_degree += degree[i]
        av_degree = sum_degree / network_size
    print("平均度为：%d" % av_degree)

    #统计度数为degree[i]的节点个数
    for i in range(network_size):
        degree_distribution[degree[i]] += 1
    print(degree_distribution)

    #计算度分布
    for i in range(network_size):
        degree_distribution[i] = degree_distribution[i] / network_size
    print(degree_distribution)

    #保存计算的度分布
    f = open("DegreeDistribution.txt", "w+")
    for i in range(network_size):
        f.write(str(i))
        f.write(" ")
        s = str(degree_distribution[i])
        f.write(str(s))
        f.write("\n")
    f.close()


if __name__ == "__main__":

    generate_adjacency_matrix()
    generate_graph()
    calculate_degree_distribution()
    write_adjacency_matrix_to_file()

end_time = time.time()
print("运行时间为：%d"%(end_time-start_time))