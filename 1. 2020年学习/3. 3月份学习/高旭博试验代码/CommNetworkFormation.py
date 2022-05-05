import numpy as np
import igraph
import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt

NETWORK_SIZE = 128
# PROBABILITY_OF_EAGE=0.8  #Limited to global
pin = 0.7
# pout=0.01
pout = 0.01

M = 8  # Community Number
Amatrix = [[0 for i in range(NETWORK_SIZE)] for i in range(NETWORK_SIZE)]
# def generateRandomNetwork()：
for i in range(0, NETWORK_SIZE):
    for j in range(i, NETWORK_SIZE):
        Amatrix[i][j] = Amatrix[j][i] = 0
# Intracommunity


intvl = int(NETWORK_SIZE / M)  #intvl：互联网
bgIntvl = 0
endIntvl = intvl

for m in range(M):
    for i in range(bgIntvl, endIntvl):
        for j in range(bgIntvl, endIntvl):
            if (i == j):
                continue
            probability = np.random.random()
            if (probability <= pin):
                Amatrix[i][j] = Amatrix[j][i] = 1
    # Update Interval
    bgIntvl += intvl
    endIntvl += intvl

# INTERcommunity
bg1 = 0
end1 = intvl
for m1 in range(M - 1):
    # Destiny Range Initial Conditions
    bg2 = end1
    end2 = end1 + intvl
    for m2 in range(M - m1 - 1):
        for i in range(bg1, end1):
            for j in range(bg2, end2):
                probability = np.random.random()
                if (probability <= pout):
                    Amatrix[i][j] = Amatrix[j][i] = 1
        # Destiny Range Update
        bg2 = end2
        end2 = end2 + intvl
    bg1 = end1
    end1 = end1 + intvl

# for i in range(NETWORK_SIZE):
#    for j in range(NETWORK_SIZE):
#        probability=np.random.random()
#        if (i==j):
#            continue
#        if(probability <= pout):
#            Amatrix[i][j] = Amatrix[j][i] = 1

# for i in range(1,20):
#    for j in range(21,NETWORK_SIZE):
#        probability=np.random.random()
#        #if(probability > PROBABILITY_OF_EAGE):
#        if(probability <= pout):
#            Amatrix[i][j] = Amatrix[j][i] = 1


G = nx.Graph()
for i in range(len(Amatrix)):
    for j in range(len(Amatrix)):
        if (Amatrix[i][j] == 1):
            G.add_edge(i, j)
nx.draw(G)
plt.show()
print(G)