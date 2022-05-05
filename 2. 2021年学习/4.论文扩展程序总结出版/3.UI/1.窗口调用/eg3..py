from PyQt5.QtWidgets import *
import sys

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.window = QMainWindow
        # self.window.resize()

        self.setWindowTitle("菜单页")   #默认继承QmainWindow所有函数，创建了一个窗口
        self.resize(1000, 400)  #设置窗口大小
        # self.window.move(700, 100) #设置窗口显示在屏幕的位置，不设置默认显示在显示屏中间

        #创建特征正提取按键
        feature = QPushButton("特征提取", self)
        feature.resize(100, 50)
        feature.move(80, 150)
        feature.clicked.connect(self.show_child)
        # self.child_window = Child()

        #second
        p = QPushButton("计算最佳P", self)
        p.resize(100, 50)
        p.move(310, 150)
        p.clicked.connect(self.show_child)
        # self.child_window = Child()

        #second
        build_net = QPushButton("形成网络", self)
        build_net.resize(100, 50)
        build_net.move(540, 150)
        build_net.clicked.connect(self.show_child)
        # self.child_window = Child()

        #second
        clacfication = QPushButton("新数据分类", self)
        clacfication.resize(100, 50)
        clacfication.move(770, 150)
        clacfication.clicked.connect(self.show_child)
        # self.child_window = Child()


    def show_child(self):
        self.child_window = Child()
        self.child_window.show()

class Child(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("统计薪资")

# 运行主窗口
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = Main()
    window.show()

    sys.exit(app.exec_())