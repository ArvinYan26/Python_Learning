#队列先进先出
class Queue(object):
    """队列"""
    def __init__(self):
        self.__list = []

    def enqueue(self, item):
        """入队， 队中添加元素"""
        self.__list.append(item)

    def dequeue(self):
        """头部删除元素"""
        return self.__list.pop(0)

    def is_empty(self):
        """判断是否为空"""
        #法一
        #if self.__list:
         #   return
        #法二
        return self.__list == []

    def size(self):
        """返回队列大小"""
        return len(self.__list)



def main():
    s = Queue()
    print(s.is_empty())
    s.enqueue(1)
    s.enqueue(2)
    s.enqueue(3)
    s.enqueue(4)
    print(s.is_empty())
    print(s.size())
    print(s.dequeue())
    print(s.size())



if __name__ == '__main__':
    main()