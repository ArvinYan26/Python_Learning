from sklearn.metrics import confusion_matrix

y_true = [2, 0, 0, 1, 2, 2, 2, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 2, 1, 0, 2, 2, 0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2, 0,
 2, 2, 1, 2, 1, 1, 2, 2]
y_predict = [2, 0, 0, 1, 2, 2, 2, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 1, 0, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2, 0,
 2, 2, 1, 0, 1, 1, 2, 0]

con_m = confusion_matrix(y_true, y_predict, labels=[0, 1, 2])
print("con_m:")
print(con_m)

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

sns.set()
f,ax=plt.subplots()
y_true = [0,0,1,2,1,2,0,2,2,0,1,1]
y_pred = [1,0,1,2,1,0,0,2,2,0,1,1]
C2= confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
print(C2)  #打印出来看看
sns.heatmap(C2, annot=True,ax=ax) #画热力图

ax.set_title('confusion matrix') #标题
ax.set_xlabel('predict') #x轴
ax.set_ylabel('true') #y轴
plt.show()