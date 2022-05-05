import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\Yan\Desktop\data1.csv')
features = list(df.columns)
"""
方法一：
features.remove('class_type')
features.remove('animal_name')
print(features)
"""
features = features[: len(features) - 1]  # 去掉开头和结尾的两列数据
# print(features)
X = df[features].values.astype(np.float32)
Y = np.array(df.class_num)
print(X, y)