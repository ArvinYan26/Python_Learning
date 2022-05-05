import numpy as np
from sklearn import preprocessing
#预处理博文链接：https://blog.csdn.net/weixin_40807247/article/details/82793220
#1.StandarScaler
# preprocessing这个模块还提供了一个实用类StandarScaler，它可以在训练数据集上做了标准转换操作之后，把相同的转换应用到测试训练集中。
# 这是相当好的一个功能。可以对训练数据，测试数据应用相同的转换，以后有新的数据进来也可以直接调用，不用再重新把数据放在一起再计算一次了。
# 调用fit方法，根据已有的训练数据创建一个标准化的转换器
# 另外，StandardScaler()中可以传入两个参数：with_mean,with_std.这两个都是布尔型的参数，
# 默认情况下都是true,但也可以自定义成false.即不要均值中心化或者不要方差规模化为1.
x = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1.]])
scaler = preprocessing.StandardScaler().fit(x)
# 使用上面这个转换器去转换训练数据x,调用transform方法
s = scaler.transform(x)
print(s)
"""
[[ 0.         -1.22474487  1.33630621]
 [ 1.22474487  0.         -0.26726124]
 [-1.22474487  1.22474487 -1.06904497]]
"""
# 好了，比如现在又来了一组新的样本，也想得到相同的转换
new_x = [[-1., 1., 0.]]
s_new = scaler.transform(new_x)
print(s_new)
#[[-2.44948974  1.22474487 -0.26726124]]


#2. # MinMaxScaler
# 在MinMaxScaler中是给定了一个明确的最大值与最小值。它的计算公式如下：
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std / (max - min) + min
# 以下这个例子是将数据规与[0,1]之间，每个特征中的最小值变成了0，最大值变成了1，请看：
#min_max_scaler = preprocessing.MinMaxScaler()
#x_minmax = min_max_scaler.fit_transform(x)

# 同样的，如果有新的测试数据进来，也想做同样的转换咋办呢？请看：
#x_test = np.array([[-3., -1., 4.]])
#x_test_minmax = min_max_scaler.transform(x_test)

#3.# 2 正则化Normalization
# 正则化是将样本在向量空间模型上的一个转换，经常被使用在分类与聚类中。
# 函数normalize 提供了一个快速有简单的方式在一个单向量上来实现这正则化的功能。
# 正则化有l1,l2等，这些都可以用上：
#x_normalized = preprocessing.normalize(x, norm='l2')

#4.Normalizer
# preprocessing这个模块还提供了一个实用类Normalizer,实用transform方法同样也可以对新的数据进行同样的转换
# 根据训练数据创建一个正则器
#normalizer = preprocessing.Normalizer().fit(x)

# 对训练数据进行正则
#normalizer.transform(x)

# 对新的测试数据进行正则
#normalizer.transform([[-1., 1., 0.]])
# normalize和Normalizer都既可以用在密集数组也可以用在稀疏矩阵（scipy.sparse)中
# 对于稀疏的输入数据，它会被转变成维亚索的稀疏行表征（具体请见scipy.sparse.csr_matrix)

list = [[5.0, 2.191666666666667, 0.8069444444444445, 0.04856293359762275, 0.7252747252747253], [4.181818181818182, 2.411255411255411, 0.5571166207529843, -0.43034435731333, 0.5814977973568282]]
list = np.array(list)
print(list)
scaler = preprocessing.MinMaxScaler()
print(scaler)
data = scaler.fit_transform(list)
print(data)