import numpy as np



class_num = 2
per_class_len = 8   #  30

def generate_dataset():
    data = [[0.43193755, 0.64956977],
            [1.63282389, 1.84777337],
            [0.78263636, 1.32217167],
            [1.10717511, 0.09804702],
            [2.29437207, 0.15075993],
            [0.01946484, 2.68549965],
            [1.65665127, 2.1076443 ],
            [2.98406233, 1.0965065 ],
            [4.82107959, 4.54081177],
            [4.85267165, 4.46316394],
            [4.63429117, 4.73252046],
            [4.2309044 , 4.75946124],
            [4.91230745, 4.59755316],
            [4.16042242, 4.88947395],
            [4.08610335, 4.06990587],
            [4.32858256, 4.07965591]]

    """
    方法一
    list = [[0 for i in range(per_class_len) for i in range(class_num)]]
    list = np.reshape(list, (class_num, per_class_len))
    print(list)
    """
    data = np.array(data)
    print(type(data))
    #方法二：
    label = np.zeros(len(data))
    #print(type(label))

    i = 0
    while i < len(data):
        if i < per_class_len:
            label[i] = 0
            i += 1
        else:
            label[i] = 1
            i += 1

    """
    print("data:", data.shape)
    print("label:", label.shape)
    print(data, label)
    """

    """
    #李根算法label
    label = np.split(label, class_num)
    #print(type(label))

    for target, i in enumerate(label):
        i[:] = target
    label = np.array(label)
    print(label)
    """
    return data, label

#data, label = generate_dataset()

