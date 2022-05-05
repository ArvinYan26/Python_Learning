class Node(object):
    """定义节点类,节点元素和下一个节点的地址"""
    def __init__(self, elem):
        self.elem = elem
        self.next = None #next：后继结点
        self.prev = None #prev:前驱结点

class DoubleLinkList(object):
    """定义单链表"""
    def __init__(self, node=None): #默认船机那里的是空节点
        self._head = node  #指向用户传进来的节点，初始化的时候是None

    def is_empty(self):
        """链表是否为空"""
        return self._head == None  #如果是None，那就是空

    def length(self):
        """链表长度，即统计元素个数"""
        # cur游标， 用来移动遍历节点, cur指向数据，二cur.next：指向下一个数据的地址
        cur = self._head
        count = 0
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
        node.next.prev = node

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
            node.prev = cur

    def insert(self, pos, item):
        """指定位置添加元素"""
        if pos <= 0: #添加到头部
            self.add(item)
        elif pos > (self.length()-1):
            self.append(item)
        else:
            cur = self._head
            count = 0
           #直接移动到插入的位置pos
            while count < (pos-1):
                count += 1
                cur = cur.next
            #当循环退出时，cur指向pos的位置
            node = Node(item)
            node.next = cur
            node.prev = cur.prev
            cur.prev.next = node
            cur.prev = node

    def remove(self, item):
        """删除一个元素"""
        cur = self._head
        while cur != None:
            if cur.elem == item:
                #先判断是不是头结点
                if cur == self._head: #如果是头结点
                    self._head = cur.next
                    if cur.next: #如果是空就不执行，只有一个结点时的情况，需要考虑prev这个前驱结点指向内容
                        cur.next.prev = None
                else: #中间结点
                    cur.prev.next = cur.next
                    if cur.next: #删除的结点时最后一个结点，先判断是否cur.next为空，不是的话在执行，说明是最后一个结点
                        cur.next.prev = cur.prev
                break #跳出循环，终止循环
            else:   #没有找到元素,就继续移动游标
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
    ll = DoubleLinkList()
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
    print(ll.travel())

    ll.insert(-1, 9)
    ll.travel()


    ll.insert(2, 10)
    ll.travel()

    print(ll.search(8))

    ll.remove(10)
    ll.travel()


if __name__ == "__main__":
    main()