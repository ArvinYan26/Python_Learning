import numpy as np
import networkx as nx
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
from sklearn.model_selection import GridSearchCV
import operator
from normalization import data_preprocess
import BuildNetwork as BN
from DrawGraph import draw_graph
from CaculateMeasures import calculate_measure
import time


class DataClassification(object):
    """iris数据集分类"""

    def __init__(self, k, num_class):
        self.k = k
        # self.g = []
        self.num_class = num_class
        self.color_map = {0: 'red', 1: 'green', 2: 'yellow'}
        self.net0_measure = []  # 存储每一类别插入Y_items中的数据时的指标
        self.impact0 = []  # storage the impact of the first class
        self.net1_measure = []
        self.impact1 = []
        self.net2_measure = []
        self.G = None
        self.predict_label = []

        # 初始化运行程序，一开始就运行
        #self.build_init_network(label=True)
        self.need_classification = []  # 计算模糊分类节点次数
        #self.single_node_insert()
        #self.accuracy()

    def fit(self, X_train, Y_train):
        """获取数据集"""
        self.X_train = X_train
        self.Y_train = Y_train

    def get_subgraph(self):
        """得到子图，并画出来"""
        num = nx.number_connected_components(self.G)
        #print("num_components:", num)
        Gcc = sorted(nx.connected_components(self.G), key=len, reverse=True)
        #list = []
        for n in range(num):
            G = self.G.subgraph(Gcc[n])
            count0 = 0
            count1 = 0
            count2 = 0
            for m in G.nodes():
                if G._node[m]["label"] == 0:
                    count0 += 1
                    if count0 > 3:  #设置阈值，查找大的组件构建初试类网络，用以后边吞并小的组件
                        self.G0 = G
                if G._node[m]["label"] == 1:
                    count1 += 1
                    if count1 > 3:
                        self.G1 = G
                if G._node[m]["label"] == 2:
                    count2 += 1
                    if count2 > 3:
                        self.G2 = G

        # 如果组件数大于类别数执行下面步骤,正常情况下，分类阶段不会用到，因为很明显的分3类
        if num > self.num_class:
            for m in range(num):
                G = self.G.subgraph(Gcc[m])
                for n in G.nodes():
                    if G._node[n]["label"] == 0:
                        count = 0
                        for a in self.G0.nodes():
                            if n == a:
                                continue
                            self.G.add_edge(str(n), str(a))
                            count += 1
                            if count == self.k:  #
                                break
                    if G._node[n]["label"] == 1:
                        count = 0
                        for a in self.G1.nodes():
                            if n == a:
                                continue
                            self.G.add_edge(str(n), str(a))
                            count += 1
                            if count == self.k:
                                break
                    if G._node[n]["label"] == 2:
                        count = 0
                        for a in self.G2.nodes():
                            if n == a:
                                continue
                            self.G.add_edge(str(n), str(a))
                            count += 1
                            if count == self.k:
                                break
    def get_params(self, deep=False):
        return {'k': self.k, 'num_class': self.num_class}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, X_test, Y_test):
        """

        :param X_items: one node inserted
        :param nodeindex: the index of the inserted node
        :param Y_items: label of the inserted node
        :return:
        """
        # print(self.X_test, self.Y_test)
        # g = self.g
        # print(len(g.nodes()))
        # insert_node_id = len(list(self.g.nodes()))
        # print(insert_node_id)
        self.G = nx.Graph()
        self.G, nbrs, radius = BN.build_init_network(self.X_train, self.Y_train,
                                                               self.G, self.k, label=True)

        # print("nodes:", self.G.nodes())
        self.get_subgraph()
        #draw_graph(self.G)
        self.get_subgraph()
        draw_graph(self.G)
        self.X_test = X_test
        self.Y_test = Y_test
        # 添加节点
        for index, instance in enumerate(self.X_test):
            # label = 4
            label = self.Y_test[index]
            print("label:", label)
            if (not len(self.Y_test) == 0):
                label = self.Y_test[index]
                # print(index, instance)
            #插入新节点的节点编号
            insert_node_id = len(list(self.G.nodes()))
            # 计算插入节点之前的各个类别网络的measures

            # self.draw_graph(self.G0)
            mf0 = calculate_measure(self.G0, insert_node_id)
            self.net0_measure.append(mf0)
            # self.draw_graph(self.G1)
            mf1 = calculate_measure(self.G1, insert_node_id)
            self.net1_measure.append(mf1)

            dist_list = []
            measures0 = []
            BN.node_insert(self.G0, self.k, nbrs, instance, insert_node_id, label)
            adj0 = [n for n in self.G.neighbors(str(insert_node_id))]
            ma0 = calculate_measure(self.G0, insert_node_id)
            V1, V2 = np.array(self.net0_measure[len(self.net0_measure) - 1]), np.array(ma0)
            measures0.append(ma0)
            euclidean_dist0 = np.linalg.norm(V2 - V1)
            dist_list.append(euclidean_dist0)

            measures1 = []
            BN.node_insert(self.G1, self.k, nbrs, instance, insert_node_id, label)
            adj1 = [n for n in self.G.neighbors(str(insert_node_id))]
            ma1 = calculate_measure(self.G1, insert_node_id)
            V1, V2 = np.array(self.net1_measure[len(self.net1_measure) - 1]), np.array(ma1)
            measures1.append(ma1)
            euclidean_dist1 = np.linalg.norm(V2 - V1)
            dist_list.append(euclidean_dist1)
            label = int(dist_list.index(min(dist_list)))
            print("classification_result:", label)
            self.predict_label.append(label)

            #查看该节点是否在图中
            if str(insert_node_id) in self.G.nodes():
                self.G.remove_node(str(insert_node_id))
            if label == 0:
                node_list = self.G0.nodes()
                neighbor = [x for x in node_list if x in adj0]
                # 然后将节点添加到类1中
                self.G.add_node(str(insert_node_id), typeNode='test', label=False)
                for n in neighbor:
                    self.G.add_edge(str(insert_node_id), n)  # add edges
            if label == 1:
                node_list = self.G1.nodes()
                neighbor = [x for x in node_list if x in adj1]
                # 然后将节点添加到类1中
                self.G.add_node(str(insert_node_id), typeNode='test', label=False)
                for n in neighbor:
                    self.G.add_edge(str(insert_node_id), n)  # add edges
            print("=" * 100)

        #得到最后的图和子图
        draw_graph(self.G)
        #self.get_subgraph()

        #得到准确率，和预测target
        label = list(map(int, self.Y_test))  # 廖雪峰，高阶函数内容
        print("original_label:", label)
        print("predict_label :", self.predict_label)

        count = 0
        for i in range(len(self.Y_test)):
            if self.Y_test[i] == self.predict_label[i]:
                count += 1
        print("正确个数：", count)
        accuracy = round(count / len(self.Y_test), 3)
        print("accuracy:", accuracy)
        print("need_classification:", self.need_classification)

        return accuracy

    #def classification(self, insert_node_id, label):


if __name__ == '__main__':

    """
    start_time = time.time()
    iris = load_iris()
    iris_data = iris.data  # [:, 2:]
    iris_target = iris.target
    acc0_list = []
    acc1_list = []
    info = []
    fina_result = []
    X_train, X_test, Y_train, Y_test = train_test_split(iris_data, iris_target, test_size=0.3)
    X_train, X_test = data_preprocess(X_train, X_test, 2)  # normType=2 归一化到0-1之间
    DC = DataClassification(k=2, num_class=3)
    DC.fit(X_train, X_test, Y_train, Y_test)
    accuracy = DC.predict(X_test, Y_test)
    print(accuracy)
    """

    """
    for k in range(2, 6):
        for j in range(5):
            for i in range(5):
                X_train, X_test, Y_train, Y_test = train_test_split(iris_data, iris_target, test_size=0.3)
                X_train, X_test = data_preprocess(X_train, X_test, 2)  # normType=2 归一化到0-1之间
                DC = DataClassification(k=k, num_class=3)
                DC.fit(X_train, X_test, Y_train, Y_test)
                accuracy = DC.predict(X_test, Y_test)
                acc0_list.append(accuracy)

        acc1_list.append(acc0_list)
        acc = np.array(acc1_list)
        print(acc)
        print("k: %d Accuracy: %0.2f (+/- %0.2f)" % (k, acc.mean(), np.std(acc) * 2))
        info.append(k)
        info.append(acc.mean())
        info.append(np.std(acc) * 2)
    fina_result.append(info)
    print(fina_result)
    """




    """
    f=open("results.txt",'w')
    f.close()
    grid_values = {'knn': range(1, 10), 'num_class': [1]}
    kfold = KFold(n_splits=5, random_state=None, shuffle=True)
    estimator = DataClassification(2, 3)
    clf = GridSearchCV(estimator, param_grid=grid_values, cv=kfold, scoring='accuracy', n_jobs=7)
    grid_result = clf.fit(iris_data, iris_target)
    print("Best Estimator: ", grid_result.best_estimator_.get_params(), ' Score: ', grid_result.best_score_)
    f=open("results.txt", 'a')
    f.write("Best Estimator: "+str(grid_result.best_estimator_.get_params())+' Score: '+str(grid_result.best_score_)+'\n')
    f.close()
    """
    """
    test = 5
    total=[]
    for i in range(test):
        DC = DataClassification(k=2, num_class=3)
        kfold = KFold(n_splits=10, random_state=None, shuffle=True)
        scores = cross_val_score(DC, X_train, Y_train, scoring="accuracy", cv=kfold)
        total.append(scores)
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    total=np.array(total)
    print("----\n->Accuracy Total: %0.2f (+/- %0.2f)" % (total.mean(), total.std() * 2))

    end_time = time.time()
    print("times:%d" % (end_time - start_time))
    print("finished")
    """