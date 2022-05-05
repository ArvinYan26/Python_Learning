from GetCOVID_19Data1 import GetData
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from build_network import *


def data_preprocess(data):
    """
    特征工程（归一化）
    """
    # 归一化
    scaler = preprocessing.MinMaxScaler().fit(data)
    data = scaler.transform(data)
    return data

if __name__ == '__main__':
    star_time = time.time()

    max_value, min_value, steps = 100, 160, 10
    gd = GetData(max_value, min_value, steps)
    # 原数据集路径
    path0 = "D:/datasets/covid-19/COVID-19-c/NORMAL/*.png"
    path1 = "D:/datasets/covid-19/COVID-19-c/COVID-19/*.png"
    data, target = gd.get_data(path0, path1)
    # print(data)
    # 数据分析
    data = data_preprocess(data)
    # single run
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1)
    # 数据分类
    DC = DataClassification(k=6, num_class=2)
    DC.fit(train_data, train_target)
    DC.predict(test_data, test_target)
    acc = DC.score()
    print("acc:", acc)

    end_time = time.time()
    run_time = end_time - star_time
    print("run_time:", run_time)