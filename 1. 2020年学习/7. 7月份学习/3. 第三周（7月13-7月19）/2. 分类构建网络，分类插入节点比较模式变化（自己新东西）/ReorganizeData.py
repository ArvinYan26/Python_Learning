import numpy as np


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
    print(len(current_data0), len(current_data1), len(current_data2))

    new_data = np.vstack((current_data0, current_data1, current_data2))
    new_target = current_data0_target + current_data1_target + current_data2_target
    #print(new_data, new_target)
    #print(len(new_data), len(new_target))

    return new_data, new_target, current_data0, current_data1, current_data2
