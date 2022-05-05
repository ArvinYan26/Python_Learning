#在pandas中有个另类的存在就是nan，解释是：not a number，
#不是一个数字，但是它的类型确是一个float类型。numpy中也存在关于nan的方法，如：np.nan
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

n = np.nan
print(type(n))

m = 1
print(n+m)  #任何数字和nan相加都是nan

#nan in Series
s1 = Series([1, 2, np.nan, 3, 4], index=['A', 'B', 'C', 'D', 'F'])
print(s1)
#np.nan:就是生成NaN浮点型的非数字
"""
A    1.0
B    2.0
C    NaN
D    3.0
F    4.0
"""
print(s1.isnull()) #返回bool值，是nan的话返回True
"""
A    False
B    False
C     True
D    False
F    False
dtype: bool
"""
print(s1.notnull()) #非控制返回True，nan的话返回False
"""
A     True
B     True
C    False
D     True
F     True
dtype: bool
"""
print(s1.dropna()) #去掉有nan的索引项
"""
A    1.0
B    2.0
D    3.0
F    4.0
dtype: float64
"""
#nan is dataframe
df = DataFrame([[1, 2, 3], [np.nan, 5, 6], [7, np.nan, 9], [np.nan, np.nan, np.nan]])
print(df)
"""
     0    1    2
0  1.0  2.0  3.0
1  NaN  5.0  6.0
2  7.0  NaN  9.0
3  NaN  NaN  NaN
"""
print(df.isnull()) #nan返回True
"""
       0      1      2
0  False  False  False
1   True  False  False
2  False   True  False
3   True   True   True
"""

