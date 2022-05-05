from PyQt5.QtWidgets import *
from feature1 import *
import sys

class Child(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("特征提取")  # 默认继承QmainWindow所有函数，创建了一个窗口
        self.resize(1000, 400)  # 设置窗口大小
        # self.window.move(700, 100) #设置窗口显示在屏幕的位置，不设置默认显示在显示屏中间

        # 创建特征正提取按键
        feature = QPushButton("灰度直方图", self)
        feature.resize(100, 50)
        feature.move(80, 150)
        feature.clicked.connect(self.show_sub_child)
        # print("按钮已经按下")
        # self.sub_child_window = SubChild()

        # second
        p = QPushButton("傅里叶变换频谱图", self)
        p.resize(100, 50)
        p.move(310, 150)
        p.clicked.connect(self.show_sub_child)
        # self.sub_child_window = SubChild()

        # second
        build_net = QPushButton("分形维数", self)
        build_net.resize(100, 50)
        build_net.move(540, 150)
        build_net.clicked.connect(self.show_sub_child)
        # self.sub_child_window = SubChild()

        # second
        clacfication = QPushButton("四叉树及像素块直方图", self)
        clacfication.resize(100, 50)
        clacfication.move(770, 150)
        clacfication.clicked.connect(self.show_sub_child)

    def show_sub_child(self):
        self.sub_child_window = SubChild()
        # print("1111111")
        self.sub_child_window.show()




