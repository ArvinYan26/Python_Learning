class Node(object):
    """节点"""
    def __init__(self, elem):
        self.elem = elem
        self.next = None

class SingleLinkList(object):
    def __init__(self, node=None): #默认参数，None，默认传进来的是空节点
        self._head = node  #开始指向用户传进来的节点

    def is_empty(self):
        """链表是否为空"""
        pass
    def legth(self):
        """链表长度"""
        pass
    def travel(self):
        """遍历整个链表"""
        pass
    def add(self, item):
        """链表头部添加属性"""
        pass
    def append(self, item):
        """链表头尾部添加属性"""
        pass
    def insert(self, pos, item):
        """指定位置添加元素"""
        pass
    def remove(self, item):
        """删除节点"""
        pass

    def search(self, item):
        """查找节点是否存在"""
        pass


node = None(100)
single_obj = SingleLinkList()
single_obj.travel()