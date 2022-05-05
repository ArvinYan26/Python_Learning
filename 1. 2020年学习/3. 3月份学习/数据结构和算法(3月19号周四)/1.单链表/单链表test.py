class Node(object):
    """定义节点类,节点元素和下一个节点的地址"""
    def __init__(self, elem):
        self.elem = elem
        self.next = None #初始化的时候还没有将节点串入链表中，所以是None

class SingleLinkList(object):
    """定义单链表"""
    def __init__(self, node=None): #默认船机那里的是空节点
        self._head = node  #指向用户传进来的节点

    def is_empty(self):
        """链表是否为空"""
        return self._head == None  #如果是None，那就是空

    def length(self):
        """链表长度，即统计元素个数"""
        # cur游标， 用来移动遍历节点, cur指向数据，二cur.next：指向下一个数据的地址
        cur = self._head
        count = 0  #之所以是0，覆盖了空链表的情况，如果是空链表，直接就是0
        while cur != None:  # 此时cur指向node这个对象，传入的是node对象中的初始换节点
            count += 1
            cur = cur.next
        return count  # 将节点数返回,即长度


    def travel(self):
        """遍历整链表,就是打印每一个元素"""
        #cur游标， 用来移动遍历节点
        cur = self._head
        #for循环没办法实现，只能while循环实现，while循环可以满足条件循环
        while cur != None: #此时cur指向node这个对象，传入的是node对象中的初始换节点
            print(cur.elem, end=" ")
            cur = cur.next
           #将节点数返回
        print(" ") #一个循环结束，

    def add(self, item):
        """链表头部添加元素"""
        node = Node(item)
        node.next = self._head #添加节点的next域先指向原来头节点的引用，即self._head,顺序不能乱，否则整个链表就会丢失
        self._head = node #然后头节点指向新节点，连接起来整个链表

    def append(self, item):
        """链表尾部添加元素"""
        #先构造节点
        node = Node(item)
        #判断链表是否为空
        if self.is_empty():
            self._head = node
        else:
            cur = self._head
            while cur.next != None:
                cur = cur.next
            cur.next = node

    def insert(self, pos, item):
        """指定位置添加元素"""
        if pos <= 0: #添加到头部
            self.add(item)
        elif pos > (self.length()-1):
            self.append(item)
        else:
            pre = self._head #定义cur前面的指针,此时count为0
            count = 0
            #pre移动一下，然后count+1，直到pre移动到pos的前一个位置停止即count=pos-1
            while count < (pos-1):
                count += 1
                pre = pre.next
            #当循环退出时，pre指向pos-1的位置
            node = Node(item)
            node.next = pre.next
            pre.next = node

    def remove(self, item):
        """删除一个元素"""
        cur = self._head
        pre = None #最开始cur指向第一个节点，而pre是前一个指针，所以指向空
        while cur != None:
            if cur.elem == item:
                #先判断是不是头节点
                if cur == self._head: #头节点
                    self._head = cur.next
                else: #中间节点
                    pre.next = cur.next
                break #找到了就跳出循环，终止
            else:   #没有找到元素,就继续移动游标
                pre = cur
                cur = cur.next

    def search(self, item):
        """查找节点是否存在"""
        cur = self._head
        while cur != None:
            if cur.elem == item:
                return True  #调用方法时，需要有一个参数去接收，否则不会自显示
            else:
                cur = cur.next
        return False #循环结束，最后也没找到，就返回False

def main():
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
    ll.travel()
    print(ll.length())

    ll.insert(-1, 9)
    ll.travel()
    print(ll.length())


    ll.insert(2, 10)
    ll.travel()
    """
    print(ll.search(8))

    ll.remove(10)
    ll.travel()
    """

if __name__ == "__main__":
    main()