"""
级联菜单栏主页模板
"""
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class MeaueDemo(QMainWindow):
    def __init__(self, parent=None):
        super(MeaueDemo, self).__init__(parent)
        bar = self.menuBar()
        file = bar.addMenu("文件")
        new = QAction("新建", file)
        new.setShortcut("Ctrl+N")
        save = QAction("保存", file)

        # 快捷设置
        save.setShortcut("Ctrl+s")
        # save.setIcon(QIcon("图形路径"))
        saveas = QAction("另存为...", file)
        saveas.setShortcut("Ctrl+Shift+s")
        # saves.setIcon(QIcon("图形路径"))   #设置快捷键
        file.addAction(new)
        file.addAction(save)
        file.addAction(saveas)
        # 添加分隔线
        line = QAction(file)
        line.setSeparator(True)
        file.addAction(line)

        # 级联菜单
        recent = file.addMenu("最近打开")
        recent.addAction("1.text")

        # 状态不可用
        prints = file.addAction("打印")
        prints.setDisabled(True)
        quit = QAction("退出", file)
        quit.setIcon(QIcon("图像路径"))
        file.addAction(quit)

        #设置功能1：特征提取， 2：计算P， 3：构建网络， 4：数据分类
        edit = bar.addMenu("特征提取")
        historgram = QAction("灰度直方图", edit)
        fft = QAction("快速傅里叶变换", edit)
        frc = QAction("分形维数", edit)
        quadtree = QAction("四叉树分割", edit)
        edit.addAction(historgram)
        edit.addAction(fft)
        edit.addAction(frc)
        edit.addAction(quadtree)
        #计算P
        par = bar.addMenu("计算网络参数")
        p = QAction("核心/边缘度量P", edit)
        epsilo = QAction("最佳距离阈值", edit)
        par.addAction(p)
        par.addAction(epsilo)

        #计算构建网络
        classfication = bar.addMenu("数据分类")
        best_net = QAction("构建最佳网络", edit)
        classficat = QAction("新数据分类", edit)
        classfication.addAction(best_net)
        classfication.addAction(classficat)


        help = bar.addMenu("帮助")
        par_help = QAction("相关参数说明", edit)

        help.addAction(par_help)

        file.triggered[QAction].connect(self.processtrigger)
        edit.triggered[QAction].connect(self.processtrigger)
        par.triggered[QAction].connect(self.processtrigger)
        edit.triggered[QAction].connect(self.processtrigger)
        help.triggered[QAction].connect(self.processtrigger)

        self.setWindowTitle("新冠肺炎诊断系统")
        self.setWindowIcon(QIcon("E:/电脑壁纸/1.jpg"))
        self.resize(1000, 800)

    def processtrigger(self, q):
        print(q.text()+"is triggered")  #单机每一个菜单按钮时


if __name__ == '__main__':
    app = QApplication(sys.argv)
    meaue = MeaueDemo()
    meaue.show()
    sys.exit(app.exec_())








