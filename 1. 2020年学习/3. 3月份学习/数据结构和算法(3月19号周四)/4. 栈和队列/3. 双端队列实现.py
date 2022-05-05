#队列先进先出
#双端队列实现
class Queue(object):
    """双端队列"""
    def __init__(self):
        self.__list = []

    def add_front(self, item):
        """往队列中添加一个item元素，入队"""
        self.__list.insert(0, item) #从头部添加元素

    def add_rear(self, item):
        """往队列中添加一个item元素"""
        self.__list.append(item)  #从尾部添加

    def pop_front(self):
        """从队头部删除一个元素，出队"""
        return self.__list.pop(0)

    def pop_rear(self):
        """从队尾部删除一个元素，出队"""
        return self.__list.pop(0)

    def is_empty(self):
        """判断一个队列是否为空"""
        return self.__list == []

    def size(self):
        """返回队列大小"""
        return len(self.__list)
if __name__ == "__main__":
    s = Queue()
    s.enqueue(1)
    s.enqueue(2)
    s.enqueue(3)
    s.enqueue(4)
    print(s.dequeue())
    print(s.dequeue())
    print(s.dequeue())
    print(s.dequeue())