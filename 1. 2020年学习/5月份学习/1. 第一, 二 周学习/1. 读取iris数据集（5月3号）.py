import numpy as np
#**问题：**导入鸢尾属植物数据集，保持文本不变。
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=', ', dtype='object')
names =  ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
print(iris, len(iris))
print("维度：", iris.ndim)
#print the first 3 rows
info = iris[:3]
print(info)

#**问题：**从前面问题中导入的一维鸢尾属植物数据集中提取文本列的物种。
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None)   #注意关键字参数的改变，之前还是object， 现在是none
#print(iris_1d)
species = np.array([row[4] for row in iris_1d])  #row[4]:每一行的第4列， species：种类
#print(species, len(species))
infos = species[:5]
print(infos)

#**问题：**通过省略鸢尾属植物数据集种类的文本字段，将一维鸢尾属植物数据集转换为二维数组iris_2d。
iris_2d = np.genfromtxt(url, delimiter=',', dtype=float, usecols=[0, 1, 2, 3, 4])
print(iris_2d)
#打印前前4行数据
iris_4 = iris_2d[:4]
print(iris_2d)
print("维度：", iris_4.ndim)


"""
#从每一个类别中取出来10个数据，构成新的数据集,然后可以根据欧式距离去计算相似度
a1 = iris_2d[:10]
#print(a1)
a2 = iris_2d[50:60]
#print(a2)
a3 = iris_2d[100:110]
#print(a3)
#合并获取的三组数组成新的数据
a = np.concatenate([a1, a2, a3], axis=0) #0:表示水平位置 
print(a)
"""

#**问题：**求出鸢尾属植物萼片长度的平均值、中位数和标准差(第1列)
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])

# Solution, mean（平均值）, medium（中位数）, Standard deviation (标准差)
mean, medium, std = np.mean(sepallength), np.median(sepallength), np.std(sepallength)
print(mean, medium, std)
# > 5.84333333333 5.8 0.825301291785

