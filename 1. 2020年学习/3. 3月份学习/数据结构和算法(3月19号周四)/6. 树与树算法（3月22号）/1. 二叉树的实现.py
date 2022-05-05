#二叉树的实现，广度优先遍历即层次遍历，横向的
#遍历方式，是类似于队列处理方式，右边添加左边取出，
#就是对链表的扩充

class Node(object):
    def __init__(self, item):
        self.elem = item
        self.lchild = None
        self.rchild = None

class Tree(object):
    """二叉树"""
    def __init__(self):  #构造函数
        self.root = None

    def add(self, item):
        node = Node(item)
        if self.root is None: #先进行判断，如果队列是空的，直接加进去元素
            self.root = node
            return
        #如果序列不为空，就加进去self.root
        queue = [self.root]
        while queue:  #对队列进行判断，是否为空，如果不为空进行循环，为空返回Flase就不会进入循环
            cur_node = queue.pop(0) #加进去的是root，然后处理完以后再取出来，放到cur_node里面进行下一步判断处理
            if cur_node.lchild is None:
                cur_node.lchild = node
                return
            else:
                queue.append(cur_node.lchild)
            if cur_node.rchild is None:
                cur_node.rchild = node
                return
            else:
                queue.append(cur_node.rchild)




tree = Tree() #