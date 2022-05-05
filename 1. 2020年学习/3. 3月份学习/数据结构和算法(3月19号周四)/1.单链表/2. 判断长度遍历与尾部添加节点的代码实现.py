class Node(object):
    """节点"""

    def __init__(self, elem):
        self.elem = elem
        self.next = None


class SingleLinkList(object):
    def __init__(self, node=None):  # 默认参数，None，默认传进来的是空节点
        self._head = node  # 开始指向用户传进来的节点

    def is_empty(self):
        """链表是否为空"""
        return self._head == None

    def length(self):
        """链表长度"""
        # cur游标，用来移动遍历节点
        cur = self._head  # 代表刚开始指向head，
        # count记录节点数量，循环实现此功能
        count = 0  # 初试为0
        while cur != None:
            count += 1
            cur = cur.next
        return count

    def travel(self):
        """遍历整个链表"""
        # cur游标，用来移动遍历节点
        cur = self._head  # 代表刚开始指向head，
        while cur != None:
            print(cur.elem, end=" ")
            cur = cur.next


    def add(self, item):
        """链表头部添加属性"""
        pass

    def append(self, item):
        """链表头尾部添加属性"""
        node = Node(item)
        if self.is_empty():
            self._head = node
        else:
            cur = self._head
            while cur.next != None:
                cur = cur.next
            cur.next = node

    def insert(self, pos, item):
        """指定位置添加元素"""
        pass

    def remove(self, item):
        """删除节点"""
        pass

    def search(self, item):
        """查找节点是否存在"""
        pass


if __name__ == "__main__":
    ll = SingleLinkList()
    print(ll.is_empty())
    print(ll.length())

    ll.append(1)
    print(ll.is_empty())
    print(ll.length())

    ll.append(2)
    ll.append(3)
    ll.append(4)
    ll.append(5)
    ll.travel()
