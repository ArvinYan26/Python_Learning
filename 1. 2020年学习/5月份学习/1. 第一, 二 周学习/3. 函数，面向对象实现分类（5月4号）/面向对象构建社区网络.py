import networkx as nx
import numpy as np
import matplotlib.pyplot  as plt

class CommunityNetworkBuild(object):
    def __init__(self, nodes, M, pin, pout):
        """
        :param nodes:网络节点数
        :param M: 社区个数
        :param M_nodes: 每个社区节点数
        :param pin: 社区内部连边概率
        :param pout: 社区间连边概率
        """
        self.nodes = nodes
        self.M = M
        self.M_nodes = self.nodes/self.M
        self. pin = pin
        self.out = pout


def main():
    CommunityNetworkBuild(128, 8, 0.7, 0.01)

if __name__ == '__main__':
    main()
