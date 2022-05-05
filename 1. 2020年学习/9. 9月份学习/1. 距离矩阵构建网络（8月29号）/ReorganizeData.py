import numpy as np

def calculate_dis(data):
    s = []
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            if not i == j:
                dis = np.linalg.norm(data[i] - data[j])
                s.append(dis)
    mean_value = np.mean(s)  #计算方差
    mean = round(mean_value, 3)
    return mean

def reorganize_data(X_train, Y_train):
    """

    :param X_train: train data
    :param Y_train: train target
    :return:
    """
    current_data0 = []
    current_data0_target = []
    current_data1 = []
    current_data1_target = []
    current_data2 = []
    current_data2_target = []

    new_data = []
    new_target = []
    for idx, instance in enumerate(X_train):
        if Y_train[idx] == 0:
            current_data0.append(instance)
            current_data0_target.append(0)

        if Y_train[idx] == 1:
            current_data1.append(instance)
            current_data1_target.append(1)
        if Y_train[idx] == 2:
            current_data2.append(instance)
            current_data2_target.append(2)
    #print(len(current_data0), len(current_data1), len(current_data2))

    new_data = np.vstack((current_data0, current_data1, current_data2))
    new_target = current_data0_target + current_data1_target + current_data2_target
    #print(new_data, new_target)
    #print(len(new_data), len(new_target))
    mean_l = []
    data_size = []
    mean0 = calculate_dis(current_data0)
    mean_l.append(mean0)
    data_size.append(len(current_data0))

    mean1 = calculate_dis(current_data1)
    mean_l.append(mean1)
    data_size.append(len(current_data1))

    mean2 = calculate_dis(current_data2)
    mean_l.append(mean2)
    data_size.append(len(current_data2))


    return new_data, new_target, mean_l, data_size
