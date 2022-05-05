import numpy as np
"""
#创建一个2*3的数组
A = np.array([[1,-1,2],[3,2,0]])
print(A)
#结果
[[ 1 -1  2]
 [ 3  2  0]]

#创建一个列向量
v = np.array([[2], [1], [3]])
print(v)
#结果
[[2]
 [1]
 [3]]

#转化为行向量，两种方法
v = np.transpose(np.array([[2,1,3]]))
print(v.T)


#求解线性方程组
A = np.array([[2,1,-2],[3,0,1],[1,1,-1]])
b = np.transpose(np.array([[-3,5,-2]]))
x = np.linalg.solve(A, b)
print(x)
"""

"""
我将用于此示例的数据集是Windsor房价数据集，其中包含有关安大略省温莎市区房屋销售的信息。 输入变量涵盖了可能对房价产生影响的一系列因素，
例如批量大小，卧室数量以及各种设施的存在。
"""

#预测房价
import csv
import numpy as np

def read_data():
    x = []
    y = []
    with open('Housing.csv') as f:
        rdr = csv.reader(f)
        # Skip the header row
        next(rdr)
        #read x and y
        for line in rdr:
            xline = [1.0]
            for s in line[:-1]:
                xline.append(float(s))
                #print(xline)
            x.append(xline)
            y.append(float(line[-1]))  #房价设为值
    return x, y

x0, y0 = read_data()
"""
# Convert all but the last 10 rows of the raw data to numpy arrays
#原始数据集包含500多个条目 为了测试线性回归模型所做预测的准确性，我们使用除最后10个数据条目之外的所有数据条目来构建回归模型并计算β。
#一旦我们构建了β向量，我们就用它来预测最后10个输入值，然后将预测的房价与数据集中的实际房价进行比较。
#最后10行数据作为测试数据
"""
d = len(x0)-10
x = np.array(x0[:d])
y = np.transpose(np.array(y0[:d]))

#computer beta
xt = np.transpose(x)
xtx = np.dot(xt, x)
xty = np.dot(xt, y)
beta = np.linalg.solve(xtx, xty)
print(beta)
# Make predictions for the last 10 rows in the data set
#对最后10行数据进行预测

for data, actual in zip(x0[d:], y0[d:]):
    x = np.array([data])
    prediction = np.dot(x, beta)
    print('prediction= '+str(prediction[0, 0]+'actual='+str(actual)))





