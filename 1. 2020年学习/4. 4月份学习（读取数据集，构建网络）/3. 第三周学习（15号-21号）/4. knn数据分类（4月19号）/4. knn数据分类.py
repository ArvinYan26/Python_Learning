from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

class KNN(object):
    """利用KNN对鸢尾花进行分类"""
    def get_iris_data(self):
        """获取数据集"""
        iris = load_iris()
        iris_data = iris.data
        iris_target = iris.target
        return iris_data, iris_target

    def run(self):
        #获取鸢尾花的特征值，目标值
        iris_data, iris_target = self.get_iris_data()
        #print(iris_data, iris_target)
        #切分数据集
        x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.25)
        #特征工程（对特征值进行归一化）
        std = StandardScaler()
        x_train = std.fit_transform(x_train)
        x_test = std.transform(x_test)

        #送入算法
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x_train, y_train)   #将测试集送入算法
        y_predict = knn.predict(x_test)  #获取预测结果

        #预测结果展示
        labels = ["山鸢尾", "虹膜锦葵", "变色鸢尾"]
        for i in range(len(y_predict)):
            print("第%d次测试：真实值：%s\t预测值：%s" %((i+1), labels[y_predict[i]], labels[y_test[i]])) #i从0开始，测试从1开始，所以要加1
        print("准确率：", knn.score(x_test, y_test))

def main():
    knn = KNN()
    knn.run()

if __name__ == "__main__":
    main()