import matplotlib.pyplot as plt
import numpy as np

l = [1, 1, 2, 3, 4, 5, 7, 8, 8, 8]
axis_x = [i for i in range(len(l))]
max = max(l)
print(max)
max_s = l.index(max)
print(max_s, max)

x = [1 for i in range(5)]
print(x)

max_x = 6
max_y = 4
max_index = l.index(max_s)
#horizontal, values = l[0:max_x+1], [max_y for i in range(max_index+1)]
horizontal, values = [max_x, max_x], [0, max_y]
print(horizontal, values)
#plt.plot(axis_x, l)
#plt.plot(horizontal, values, 'r')
plt.plot([max_x, max_x], [0, max_y], 'r--', label='Highest Value')
print([min(x), max_x], [max_y, max_y])
plt.plot([min(x), max_x], [max_y, max_y], 'r--')

"""
plt.plot(horizontal, values,'r--', label='Highest Value')
plt.plot([min(x), max_x], [max_y, max_y], 'r--')
plt.text(max_x, 0, str(max_x), fontsize='x-large')
plt.text(min(x), np.mean(max_y), str(np.mean(max_y)), fontsize='x-large')
plt.legend(loc='best')
"""

plt.xlabel("Threshold")
plt.ylabel("Measure Value")
plt.grid(True, linestyle="--", color="g", linewidth="0.5")
plt.show()

A = np.ones((5, 5))
A[2][3] = 5
print(A)
for i in range(5):
    if sum(A[i]) == 9:
        print(i, A[i])