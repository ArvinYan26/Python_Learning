import numpy as np

x = np.array([])
print(x.shape, x)
y = np.array(3)
print(y)

z = np.array(3)
print(z)

new = np.hstack((x, y, z))
print(new)
print(new.shape)

new_m = np.vstack((x, y, z))
print(new_m.shape, new_m)