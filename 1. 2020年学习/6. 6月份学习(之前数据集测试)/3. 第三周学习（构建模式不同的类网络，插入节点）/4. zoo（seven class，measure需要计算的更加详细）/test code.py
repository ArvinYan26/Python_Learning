import numpy as np
a = np.array([True, False])
print(a)
a = a + 0
print(a)


import pandas as pd

def get_data():
    df = pd.read_csv('zoo.csv')
    features = list(df.columns)

    """
    方法一：
    features.remove('class_type')
    features.remove('animal_name')
    print(features)
    """
    features = features[1 : len(features)-1]  #去掉开头和结尾的两列数据
    print(features)
    X = df[features].values.astype(np.float32)
    Y = np.array(df.class_type)


get_data()