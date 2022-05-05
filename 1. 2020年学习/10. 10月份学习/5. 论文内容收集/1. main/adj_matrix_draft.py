import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def draw_adj_matrix(adj_matrix, c_n):
    m = np.zeros_like(adj_matrix) - 2
    size = adj_matrix.shape[0]
    m[:c_n, :c_n] = 0
    m[:c_n, c_n:] = 1
    m[c_n:, :c_n] = 1

    for i in range(size):
        m[i, i] = -1
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ['white', '#000000', '#6495ED', '#FF6A6A']
    # ax.matshow(m, cmap=plt.cm.Blues)
    cmap = mpl.colors.ListedColormap(colors)
    ax.matshow(m, cmap=cmap)

    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            v = adj_matrix[j, i]
            ax.text(i, j, str(v), va='center', ha='center')

    plt.show()



intersection_matrix = np.random.randint(0, 2, size=(30, 30))

draw_adj_matrix(intersection_matrix, 10)
