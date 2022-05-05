#二叉树的实现，广度优先遍历即层次遍历，横向的
#遍历方式，是类似于队列处理方式，右边添加左边取出，
#就是对链表的扩充

#先中后指的都是根的顺序是先中后
#先序：根 - 左子数 - 右子树
#中序：左子数 - 根 - 右子树
#后序：左子树 - 右子树 - 根

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

    def breadth_travel(self):
        """广度遍历"""
        if self.root is None:
            return
        queue = [self.root]
        while queue:  #对队列进行判断，是否为空，如果不为空进行循环，为空返回Flase就不会进入循环
            cur_node = queue.pop(0) #加进去的是root，然后处理完以后再取出来，放到cur_node里面进行下一步判断处理
            print(cur_node.elem, end =" ")
            if cur_node.lchild is not None:
                queue.append(cur_node.lchild)
            if cur_node.rchild is not None:
                queue.append(cur_node.rchild)

    def preorder(self, node):
        """先序遍历"""
        if node is None: #传入的是叶子
            return
        print(node.elem, end=" ")
        self.preorder(node.lchild)
        self.preorder(node.rchild)

    def inorder(self, node):
        """中序遍历"""
        if node is None: #传入的是叶子
            return
        self.inorder(node.lchild)
        print(node.elem, end=" ")
        self.inorder(node.rchild)

    def postorder(self, node):
        """先序遍历"""
        if node is None:  # 传入的是叶子
            return
        self.postorder(node.lchild)
        self.postorder(node.rchild)
        print(node.elem, end=" ")

if __name__ == "__main__":
    tree = Tree()
    tree.add(0)
    tree.add(1)
    tree.add(2)
    tree.add(3)
    tree.add(4)
    tree.add(5)
    tree.add(6)
    tree.add(7)
    tree.add(8)
    tree.add(9)
    tree.breadth_travel()
    print(" ")
    tree.preorder(tree.root)
    print(" ")
    tree.inorder(tree.root)
    print(" ")
    tree.postorder(tree.root)