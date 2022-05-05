class Node(object):
    """节点"""
    def __init__(self, elem):
        self.elem = elem
        self.next = None #节点初始化是空的，所以指向None

class SingleCycleLinkList(object):
    """单向循环列表"""
    def __init__(self, node=None): #初始化传进来的是空节点
        self._head = node
        if node:
            node.next = node #有节点了，节点下一个区域指向节点头部，因为是循环列表

    def is_empty(self):
        """判断是否为空"""
        return self._head == None

    def length(self):
        """判断长度，因为终止条件不同于单链表，所以需要分情况讨论"""
        if self.is_empty():
            return 0
        #定义游标
        cur = self._head
        count = 1 #初始化为1，是因为单链表中，cur可以指向最后的None，而单向循环列表cur只能指向最后的节点，count少数一个，所以最开始从1数
        while cur.next != self._head: #因为当最后一个节点时，它的next区域指向self._head.这与单链表的结束判断不同，因为是循环的，
                                # 单链表只需要判断，！= None即可
            count += 1
            cur = cur.next
        return count

    def travel(self):
        """遍历整个列表元素，即打印每一个每一个元素"""
        if self.is_empty():
            return
        cur = self._head
        while cur.next != self._head:
            print(cur.elem, end=" ")
            cur = cur.next
        #循环终止的时候，尾节点元素未打印，所以需要再打印
        print(cur.elem)  #专门用来打印尾节点，因为单链表cur可以移到最后None,所以可以不用专门打印尾节点的语句


    def add(self, item):
        """链表头部添加元素, 头插法"""
        node = Node(item)
        if self.is_empty():
            self._head = node
            node.next = node #指向自己
        else:
            #下面的三行循环体是为了让游标cur找到最后的尾节点引用。然后最后指向插入的头节点，完成列表循环引用
            cur = self._head
            while cur.next != self._head:
                cur = cur.next
            #退出循环后，cur指向尾节点
            node.next = self._head
            self.head = node
            cur.next = node  #循环结束后指向为节点的cur，让他指向node，完成单向循环列表的地址循环指引。画图自己琢磨

    def append(self, item):
        """链表尾部添加元素, 尾插法"""
        node = Node(item)
        if self.is_empty():
            self._head = node
            node.next = node
        else:
            cur = self._head
            while cur.next != self._head:
                cur = cur.next
            node.next = self._head
            cur.next = node

    def insert(self, pos, item):  #insert(2, 100)
        """指定位置添加元素
        ：param:  pos , start at 0
        """
        if pos <= 0:
            self.add(item)
        elif pos > (self.length()-1):
            self.append(item)
        else:
            #pre是移动到当前pos位置的游标，而此处我们必须要移动到pos-1的位置，所以用pre游标表示更能清楚的知道位置信息，不容易混淆
            pre = self._head
            count = 0
            while count < (pos-1): #pos-1：是要插入位置的前一个元素位置
                count += 1
                pre = pre.next #用count控制pre移动的次数
            #循环结束后，pre指向要添加的位置的额前一个元素

            node = Node(item)
            node.next = pre.next
            pre.next = node

    def remove(self, item):
        """删除节点"""
        #先判断是否为空
        if self.is_empty():
            return
        #非空
        cur = self._head #因为此时的cur指向的是要被删除的元素
        pre = None
        while cur.next != self._head: #移动游标，不到最后尾节点不退出循环
            if cur.elem == item: #判断是否已经找到此元素
                #先判断此节点是否为头节点
                if cur == self._head: #找头节点
                    #要删除头节点
                    #先找尾节点
                    rear = self._head # rear:尾节点位置
                    while rear.next != self._head:
                        rear = rear.next
                    #循环结束，rear指向尾节点
                    #变换指引就可以删除节点
                    self._head = cur.next
                    rear.next = self._head
                else:
                    #中间节点删除
                    pre.next = cur.next
                break #完成循环直接跳出来终止循环
            else:
                # 不是头部节点，中间节点没找到，然后移动游标，继续找该节点
                pre = cur
                cur = cur.next
        #尾节点,退出循环后，cur指向尾节点，删除尾节点
        if cur.elem == item:
            pre.next = cur.next


    def search(self, item):
        """查找节点是否存在"""
        if self.is_empty():
            return False
        cur = self._head
        while cur.next != self._head: #判断尾节点的指向的区域是不是头节点，不是的话就继续循环
            if cur.elem == item:
                return True
            else:
                cur = cur.next
        # 退出循环，cur指向尾节点，因为条件是cur.next,但是尾节点是否是要查找的节点还没有进行判断
        # 判断尾节点是不是查找的元素
        if cur.elem == item:
            return True
        # 所有循环结束，没有要查找的元素，返回False
        return False


def main():
    ll = SingleCycleLinkList()
    #print(ll.is_empty())
    #print(ll.length())

    ll.append(1)
    ll.append(2)
    ll.append(8)
    ll.append(3)
    ll.append(4)
    ll.append(5)
    ll.travel()
    print(ll.is_empty())
    print(ll.length())
    """
    ll.append(2)
    ll.append(8)
    ll.append(3)
    ll.append(4)
    ll.append(5)
    # 1 2 8 3 4 5 6
    ll.insert(-1, 9)  # 添加到最头部
    ll.travel()
    ll.insert(2, 100)  # 索引为2位置添加100这个元素
    ll.travel()
    ll.insert(10, 200)
    ll.remove(100)
    ll.travel()
    ll.remove(9)
    ll.travel()
    ll.remove(200)
    ll.travel()
    """
if __name__ == "__main__":
    main()