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
        print(" ")

    # 时间复杂度O（1）
    def add(self, item):
        """链表头部添加属性"""
        node = Node(item)
        node.next = self._head
        self._head = node

    # 时间复杂度O（n）
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

    # 时间复杂度O（n）
    def insert(self, pos, item): #pos:位置参数
        """指定位置添加元素"""
        if pos <= 0:  #头部添加元素
            self.add(item)
        elif pos > (self.length()-1):
            self.append(item)
        else:
            pre = self._head  #pre是指向的cur的前一个元素的游标，pre指向第一个节点位置
            count = 0
            while count < (pos-1):
                count += 1
                pre = pre.next
            #当循环退出后，pre指向pos-1位置
            node = Node(item)
            node.next = pre.next
            pre.next = node


    def remove(self, item):
        """删除节点"""
        cur = self._head
        pre = None
        while cur != None:
            if cur.elem == item:
                #先判断此节点是否为头节点
                #头节点
                if cur == self._head:
                    self._head = cur.next
                else:
                   pre.next = cur.next
                break
            else:
                pre = cur
                cur = cur.next

    def search(self, item):
        """查找节点是否存在"""
        cur = self._head
        while cur != None:
            if cur.elem == item:
                return True
            else:
                cur = cur.next
        return False


if __name__ == "__main__":
    ll = SingleLinkList()
    print(ll.is_empty())
    print(ll.length())

    ll.append(1)
    print(ll.is_empty())
    print(ll.length())

    ll.append(2)
    ll.append(8)
    ll.append(3)
    ll.append(4)
    ll.append(5)
    #1 2 8 3 4 5 6
    ll.insert(-1, 9) #添加到最头部
    ll.travel()
    ll.insert(2, 100) #索引为2位置添加100这个元素
    ll.travel()
    ll.insert(10, 200)
    ll.remove(100)
    ll.travel()
    ll.remove(9)
    ll.travel()
    ll.remove(200)
    ll.travel()
