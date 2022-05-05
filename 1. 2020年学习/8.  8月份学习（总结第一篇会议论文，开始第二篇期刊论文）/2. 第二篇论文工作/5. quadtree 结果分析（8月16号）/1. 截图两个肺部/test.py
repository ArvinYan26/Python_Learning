import numpy as np


a = [x for x in range(10)]
b = [x for x in range(10, 20)]
c = [x for x in range(15, 25)]

d = []
d. append(a)
d.append(b)
d.append(c)
print(d)

v = []
for i in d:
    x = []
    for j in i:
        per = round(j / sum(i), 3)
        x.append(per)
    v.append(x)
print(np.array(v))

C = Classification(self.G, self.num_class, self.X_train, self.nbrs, insert_node_id,
                   self.net0_measure, self.net1_measure, self.net2_measure)
self.result, self.need_classification = C.classification(self.result, self.need_classification)
