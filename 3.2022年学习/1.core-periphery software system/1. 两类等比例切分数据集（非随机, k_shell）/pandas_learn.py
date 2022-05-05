import pandas as pd
import numpy as np


# a = [1, 2, 3]
# b = [4, 5, 6]
# dataframe = pd.DataFrame({"a":a, "b":b})
# dataframe.to_csv("test.csv", index=False, sep=",")



a = [['2', '1.2', '4.2', '4', '5', '7', '0'], ['0', '10', '0.3', '4', '5', '7', '1'], ['1', '5', '0', '4', '5', '7', '2']]
a = np.array(a)
print(a)
df = pd.DataFrame(a, columns=['100', '110', '120', '130', '140', '150', 'target'])
print(df)
df.to_csv("E:/PycharmProjects/1.Python Fundamental Programme/2. 2021年学习/4.论文扩展程序总结出版/data.csv", sep=",")

