from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton,  QPlainTextEdit, QMessageBox
from PySide2.QtGui import QIcon
#QMainWindow、QPlainTextEdit、QPushButton 是3个控件类，分别对应界面的主窗口、文本框、按钮.他们都是控件基类对象QWidget的子类。
"""
工资表：姓名，薪资，年龄
薛蟠     4560 25
薛蝌     4460 25
薛宝钗   35776 23
薛宝琴   14346 18
王夫人   43360 45
王熙凤   24460 25
王子腾   55660 45
王仁     15034 65
尤二姐   5324 24
贾芹     5663 25
贾兰     13443 35
贾芸     4522 25
尤三姐   5905 22
贾珍     54603 35
"""
class Stats():
    def __init__(self):
        self.window = QMainWindow()
        self.window.resize(1000, 800)
        self.window.move(700, 100)
        self.window.setWindowTitle('薪资统计')

        self.textEdit = QPlainTextEdit(self.window)
        self.textEdit.setPlaceholderText("请输入薪资表")
        self.textEdit.move(10, 25)
        self.textEdit.resize(500, 400)

        self.button = QPushButton('特征提取', self.window)
        self.button.move(700, 40)

        # 另外添加的窗口
        self.button2 = QPushButton('计算measures', self.window)
        self.button2.move(700, 100)
        self.button3 = QPushButton('构建训练网络', self.window)
        self.button3.move(700, 160)
        self.button3 = QPushButton('分类', self.window)
        self.button3.move(700, 220)

        self.button.clicked.connect(self.handleCalc)


    def handleCalc(self):
        info = self.textEdit.toPlainText()

        # 薪资20000 以上 和 以下 的人员名单
        salary_above_20k = ''
        salary_below_20k = ''
        for line in info.splitlines():
            if not line.strip():
                continue
            parts = line.split(' ')
            # 去掉列表中的空字符串内容
            parts = [p for p in parts if p]
            name, salary, age = parts
            if int(salary) >= 20000:
                salary_above_20k += name + '\n'
            else:
                salary_below_20k += name + '\n'

        QMessageBox.about(self.window,
                    '统计结果',
                    f'''薪资20000 以上的有：\n{salary_above_20k}
                    \n薪资20000 以下的有：\n{salary_below_20k}'''
                    )

if __name__ == '__main__':
    app = QApplication([])
    app.setWindowIcon(QIcon("E:/电脑壁纸/1.jpg"))  #将图片位置放进来，设置程序图标，
    stats = Stats()
    stats.window.show()
    app.exec_()


