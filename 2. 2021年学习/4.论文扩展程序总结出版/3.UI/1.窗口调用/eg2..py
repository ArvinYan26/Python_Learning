from PyQt5.QtWidgets import *
import sys

class Main(QMainWindow):
    def __init__(self):
        super().__init__()   #单继承父类
        self.setWindowTitle("主窗口")
        button = QPushButton("弹出子窗", self)     #添加按钮，
        button.clicked.connect(self.show_child)  #并把按钮信号关联槽，在槽函数中调用子窗口对象的 show 方法
        self.child_window = Child() #主窗口__init__方法中创建子窗口对象并赋值为对象属性

    def show_child(self):
        self.child_window.show()

class Child(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("我是子窗口啊")

# 运行主窗口
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = Main()
    window.show()

    sys.exit(app.exec_())