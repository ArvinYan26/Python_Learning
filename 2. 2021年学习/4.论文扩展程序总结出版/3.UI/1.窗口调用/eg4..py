from PyQt5.QtWidgets import *
import sys

# 这里涉及到一个概念 模式对话框 与 非模式对话框 （modeless dialog | modal dialog）
# 模式对话框，就是在弹出窗口的时候，整个程序就被锁定了，处于等待状态，直到对话框被关闭。这时往往是需要对话框的
# 返回值进行下面的操作。如：确认窗口（选择“是”或“否”）。
# 非模式对话框，在调用弹出窗口之后，调用即刻返回，继续下面的操作。这里只是一个调用指令的发出，不等待也不做任何处理。如：查找框。
# show() ------  modeless dialog
# exec() ------- modal dialog

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("主窗口")
        button = QPushButton("弹出子窗", self)
        button.clicked.connect(self.show_child)

    def show_child(self):
        child_window = Child()
        child_window.exec()  #

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


