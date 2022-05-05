#产生10个1-50之间的随机数，添加到列表中
import random



"""
import numpy as np
import random

#!/usr/bin/python
# -*- coding: UTF-8 -*-
import random

print( random.randint(1,10) )        # 产生 1 到 10 的一个整数型随机数  
print( random.random() )             # 产生 0 到 1 之间的随机浮点数
print( random.uniform(1.1,5.4) )     # 产生  1.1 到 5.4 之间的随机浮点数，区间可以不是整数
print( random.choice('tomorrow') )   # 从序列中随机选取一个元素
print( random.randrange(1,100,2) )   # 生成从1到100的间隔为2的随机整数

a=[1,3,5,6,7]                # 将序列a中的元素顺序打乱
random.shuffle(a)
print(a)

random_list = []
for i in range(10):
    rand = random.randint(1, 50)
    random_list.append(rand)

print(random_list)
"""

#产生10个不同的随机数
#如果相同就重新产生，如果用for循环，个数不一定是10个，所以用while循环
random_list = []
i = 0
while i < 10:
    rand = random.randint(1, 20)
    if rand not in random_list:
        random_list.append(rand)
        i += 1
print(random_list)


print("---------自定义求最大值-----------")
#假设列表中的第一个元素为最大值
max_value = random_list[0]
min_value = random_list[0]
for value in random_list:
    if value > max_value:
        max_value = value
    if value < min_value:
        min_value = value
print("最大值是：", max_value, "最小值是：", min_value)

print("---------自定义求列表元素和-----------")
sum = 0
for value in random_list:
    sum += value
print("列表元素和是：", sum)

"""
#系统求和
sum1 = sum(random_list)
print("系统求和结果是：", sum1)
"""