import numpy as np
import igraph
from igraph import Graph, mean
import networkx as nx
import random
# from random import random
import scipy as sp
import matplotlib.pyplot as plt
import math


def GenerateAdjMatrix(NETWORK_SIZE):
    Amatrix = [[0 for i in range(NETWORK_SIZE)] for i in range(NETWORK_SIZE)]
    # def generateRandomNetwork()閿?
    for i in range(0, NETWORK_SIZE):
        for j in range(i, NETWORK_SIZE):
            Amatrix[i][j] = Amatrix[j][i] = 0
    return Amatrix


def GraphFromMatrix(Amatrix):
    G = nx.Graph()
    # Connection Creator
    for i in range(len(Amatrix)):
        for j in range(len(Amatrix)):
            if (Amatrix[i][j] == 1):
                G.add_edge(i, j)
    print(nx.is_connected(G))
    degrees = G.degree()
    print(mean(degrees.values()))
    sum_of_edges = sum(degrees.values())
    # print(ztotal)

    # Printing
    plt.figure()
    pos = nx.spring_layout(G)
    for m in range(M):
        nx.draw_networkx_nodes(G, pos,
                               nodelist=[i for i in range(m * intvl, (m + 1) * intvl)],
                               node_color='b')

    nx.draw_networkx_edges(G, pos)
    plt.show()
    # print(G)

    return G


def CommunityStructuredNetwork(Amatrix):
    zin = 0
    zout = 0
    ztotal = 0

    NETWORK_SIZE = len(Amatrix)
    intvl = int(NETWORK_SIZE / M)

    # INTRAcommunity

    # for i in range(bgIntvl,endIntvl):
    #   for j in range(bgIntvl,endIntvl):
    while (ztotal * 2 / NETWORK_SIZE < K):
        # intra-community
        if (zin * 2 / NETWORK_SIZE < 15):
            for m in range(M):
                v1 = random.randint(m * intvl, (m + 1) * intvl - 1)
                v2 = random.randint(m * intvl, (m + 1) * intvl - 1)

                if (v1 == v2):
                    continue
                # if (zin/NETWORK_SIZE< K):
                probability = np.random.random()
                if (probability <= pin):
                    if (Amatrix[v1][v2] == 0):
                        Amatrix[v1][v2] = Amatrix[v2][v1] = 1
                        zin = zin + 1

        # inter-community
        if (zout * 2 / NETWORK_SIZE < 2):
            keepWalking = True
            while (keepWalking):
                v3 = random.randint(0, NETWORK_SIZE - 1)
                v4 = random.randint(0, NETWORK_SIZE - 1)
                if (math.floor(v3 / intvl) != math.floor(v4 / intvl)):  # Same community?
                    keepWalking = False  # If so, move on (If not, choose new 2-random nodes: keep on While)
            probability = np.random.random()
            if (probability <= pout):
                if (Amatrix[v3][v4] == 0):
                    Amatrix[v3][v4] = Amatrix[v4][v3] = 1
                    zout = zout + 1
        ztotal = zin + zout

    return Amatrix


def CombineMatrix(MatrixList):
    quantity = len(MatrixList)
    old_size = len(MatrixList[0])
    new_size = quantity * len(MatrixList[0])

    combinedMatrix = [[0 for i in range(new_size)] for i in range(new_size)]
    incrementFactor = 0

    for n in range(quantity):
        actualMatrix = MatrixList[n]
        incrementFactor = n * old_size

        for i in range(0, old_size):
            for j in range(0, old_size):
                combinedMatrix[i + incrementFactor][j + incrementFactor] = actualMatrix[i][j]

    return combinedMatrix


def TopCommunity(Amatrix):
    # zin = M*len(Amatrix) # ATTENTION
    # zin = K_old*NETWORK_SIZE/2 # old ztotal (previous round)
    zout = 0
    # ztotal = 0
    K3 = 1

    # inter-community
    while (zout * 2 / NETWORK_SIZE < K3):
        # if(ztotal*2/NETWORK_SIZE<K3):
        keepWalking = True
        while (keepWalking):
            v3 = random.randint(0, NETWORK_SIZE - 1)
            v4 = random.randint(0, NETWORK_SIZE - 1)
            if (math.floor(v3 / intvl) != math.floor(v4 / intvl)):  # Same community?
                keepWalking = False  # If so, move on (If not, choose new 2-random nodes: keep on While)
        probability = np.random.random()
        if (probability <= pout):
            if (Amatrix[v3][v4] == 0):
                Amatrix[v3][v4] = Amatrix[v4][v3] = 1
                zout = zout + 1
        # ztotal= ztotal + zout

    return Amatrix


NETWORK_SIZE = 48
# PROBABILITY_OF_EAGE=0.8  #Limited to global
pout = 0.2
pin = 0.8
# pout=0.01

zin = 0
zout = 0
ztotal = 0
K = 17
M = 3  # Community Number
g = int(NETWORK_SIZE / M)  # Community Size

Amatrix1 = GenerateAdjMatrix(NETWORK_SIZE)
Amatrix2 = GenerateAdjMatrix(NETWORK_SIZE)
Amatrix3 = GenerateAdjMatrix(NETWORK_SIZE)
# Amatrix4 = GenerateAdjMatrix(NETWORK_SIZE)

intvl = int(NETWORK_SIZE / M)

CommunityStructuredNetwork(Amatrix1)
CommunityStructuredNetwork(Amatrix2)
CommunityStructuredNetwork(Amatrix3)
# CommunityStructuredNetwork(Amatrix4)


# Combining Matrices
MatrixList = [Amatrix1, Amatrix2, Amatrix3]
AmatrixTop = CombineMatrix(MatrixList)

# TOP Community
# Newer Parameters
NETWORK_SIZE = NETWORK_SIZE * 3
intvl = int(NETWORK_SIZE / M)
# pout=pout/2
# K_old = K
# K = K+1
A3 = TopCommunity(AmatrixTop)

standard_community = [0] * 3
for m in range(M):
    standard_community[m] = [i for i in range(m * intvl, (m + 1) * intvl)]

A3 = np.array(A3)


class SilvaModelHierarchy:

    def __init__(self,
                 A: np.ndarray, K: int,
                 # where I modify
                 lambd: float = 0.2,
                 # modify Delta
                 Delta: float = 0.1,
                 epsilon: float = 0.05,
                 omega: (float, float) = (0, 1)):

        assert A.ndim == 2
        assert A.shape[0] == A.shape[1]
        # where I modify
        assert K >= 1

        assert lambd >= 0
        assert Delta >= 0
        assert epsilon >= 0
        assert len(omega) == 2 and omega[0] < omega[1]

        self._A = A
        self._V = A.shape[0]  # number of vertices
        self._K = K
        self._lambd = lambd
        self._Delta = Delta
        self._epsilon = epsilon
        self._omega = omega

        # Initialize the position of each particle.
        self._p = np.random.randint(self._V, size=self._K)
        # print(self._p)

        self._Prand = A / A.sum(axis=1, keepdims=True)
        # print(self._Prand)

        self._N = np.ones((self._V, self._K))
        for k in range(self._K):
            self._N[self._p[k], k] = 2
        # print(self._N)

        # Calculate the dominance value.
        self._Nbar = self._N / self._N.sum(axis=1, keepdims=True)
        # print(self._Nbar)

        self._Nbar_diff_norm = np.inf

        self._E = np.full(self._K, self._omega[0] +
                          (self._omega[1] - self._omega[0]) / self._K)
        # print(self._E)

    def update_Particle(self, new_K):
        self.K = new_K
        # self._p extend (maintain old values)

        # Initialize the position of each particle.
        # self._p = np.random.randint(self._V, size=self._K)

    def set_lambd(self, l):
        self._lambd = l

    def iterate(self):

        next_position = np.zeros(self._K, dtype=int)
        # print(self._E)

        for k in range(self._K):
            Ppref = 1.0 * self._A
            for j in range(self._V):
                Ppref[:, j] = Ppref[:, j] * self._Nbar[j, k]
            Ppref = np.copy(Ppref) / Ppref.sum(axis=1, keepdims=True)

            Prean = 1.0 * np.zeros((self._V, self._V))
            # XXX: this is an implementation detail. It is not described in the
            # paper.
            dominated = np.where(
                self._Nbar[:, k] == np.max(self._Nbar, axis=1))
            for u in dominated:
                Prean[:, u] = 1.0
            Prean = np.copy(Prean) / Prean.sum(axis=1, keepdims=True)

            S = 0 if self._E[k] > self._omega[0] else 1

            P = (1 - S) * (self._lambd * Ppref + (1 - self._lambd) *
                           self._Prand) + S * Prean

            next_position[k] = np.random.choice(self._V, p=P[self._p[k], :])
            # print("particle {} went from {} to {}.".format(k, self._p[k],
            #                                                next_position[k]))

        for k in range(self._K):
            self._N[next_position[k], k] = self._N[next_position[k], k] + 1
            self._p[k] = next_position[k]

            is_dominated = self._Nbar[next_position[k], k] == np.max(
                self._Nbar[next_position[k], :])

            self._E[k] = np.clip(self._E[k] +
                                 (1.0 if is_dominated else -
                                 1.0) *
                                 self._Delta, self._omega[0], self._omega[1])

        next_Nbar = self._N / self._N.sum(axis=1, keepdims=True)
        self._Nbar_diff_norm = np.linalg.norm(next_Nbar - self._Nbar,
                                              ord=np.inf)
        self._Nbar = next_Nbar

        # print(self._Nbar_diff_norm)

    def has_converged(self) -> bool:
        return self._Nbar_diff_norm < self._epsilon

    def result(self) -> np.ndarray:
        return np.copy(self._Nbar)


def Time_list(Interval, length):
    data_list = []
    for i in range(length):
        if ((i + 1) % Interval == 0):
            data_list.append(i + 1)
    return data_list


model1 = SilvaModelHierarchy(A3, 9)
model1.__init__(A3, 9)

for t in range(200):
    model1.iterate()

model1.iterate()