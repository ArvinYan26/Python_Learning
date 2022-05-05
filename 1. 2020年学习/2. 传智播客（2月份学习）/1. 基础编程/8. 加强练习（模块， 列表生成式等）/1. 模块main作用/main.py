#第一种导入方式
from sendmsg import test1
import sendmsg
import recvmsg
sendmsg.test1()
sendmsg.test2()
recvmsg.test2()
test1()

#第二种导入方式
#from sendmsg import test1
#test1()

"""
#如果里面有多个函数,用下面语句全部导入,但是尽量少用
from sendmsg import *
from recvmsg import *

test1()
test2() #两个包里都有teste2这个函数，但是后来调用的会覆盖前边调用的，直接用import导入会避免这种情况出现
"""
#第三种方式，导入的模块名字特别长就用as
import networkx as nx

####注意：不要起自己的程序文件名和模块名相同，这样没在你导入模块时可能会导入你自己的文件，导致无法运行程序
#实际开发中，用模块，可以方便团队分配任务，每个人开发不同模块的功能，同时模块之间功能不同的话，也会降低程序之间的耦合性，不至于动一处，全都得改。

