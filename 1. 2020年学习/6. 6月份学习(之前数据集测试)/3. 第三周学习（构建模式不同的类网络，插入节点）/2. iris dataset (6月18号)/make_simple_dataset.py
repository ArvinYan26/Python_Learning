import numpy as np



class_num = 3
per_class_len = 8
data_len = class_num*per_class_len   #  30

def generate_dataset():
    data = []
    for i in range(class_num):
        values = np.random.rand(per_class_len, 2) + i*2
        data.append(values)
    data = np.array(data).reshape(data_len, 2)

    """
    方法一
    list = [[0 for i in range(per_class_len) for i in range(class_num)]]
    list = np.reshape(list, (class_num, per_class_len))
    print(list)
    """

    #方法二：
    label = np.zeros(data_len)
    #print(type(label))

    i = 0
    while i < data_len:
        if i < per_class_len:
            label[i] = 0
            i += 1
        elif per_class_len <= i < per_class_len*2:
            label[i] = 1
            i += 1
        else:
            label[i] = 2
            i += 1
    """
    print("data:", data.shape)
    print("label:", label.shape)
    print(data, label)


    
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

