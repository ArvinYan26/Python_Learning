# 1 弥补缺失数据 #sklearn链接：https://scikit-learn.org/stable/modules/impute.html#impute
# 在scikit-learn的模型中都是假设输入的数据是数值型的，并且都是有意义的，如果有缺失数据是通过NAN，或者空值表示的话，就无法识别与计算了。
# 要弥补缺失值，可以使用均值，中位数，众数等等。Imputer这个类可以实现。请看：
import numpy as np
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')  #用均值mean替换缺失值np.nan
imp.fit([[1, 2], [np.nan, 3], [7, 6]])   #训练数据
x = [[np.nan, 2], [6, np.nan], [7, 6]]
x_new = imp.transform(x)     #弥补缺失的值
print(x_new)

#当使用“ most_frequent”或“ constant”策略时，SimpleImputer类还支持表示为字符串值或pandas分类的分类数据：

import pandas as pd
df = pd.DataFrame([["a", "x"],
                   [np.nan, "y"],
                   ["a", np.nan],
                   ["b", "y"]], dtype="category")
imp = SimpleImputer(strategy="most_frequent")  #用每一列的众数代替每一列的缺失值
print(imp.fit_transform(df))
