#顺序表写栈，后进先出，堆栈操作是压栈(运用列表知识写)
class Stack(object):
    def __init__(self):
        self.__list = [] #私有的

    def push(self, item):
        """添加新元素，压栈"""
        self.__list.append(item)

    def pop(self):
        """弹出栈顶元素，取出来元素"""
        return self.__list.pop()

    def peek(self):
        """返回栈顶元素"""
        #先判断是否为空
        if self.__list:
            return self.__list[-1]
        else:
            return None

    def is_empty(self):
        """判断栈是否为空"""
        return self.__list == []

    def size(self):
        """返回栈的元素个数"""
        return len(self.__list)



def main():
    s = Stack()
    print(s.is_empty())
    s.push(1)
    s.push(2)
    s.push(3)
    s.push(4)
    s.push(5)
    print(s.size())
    print(s.pop())
    print(s.size())
    print(s.is_empty())


if __name__ == '__main__':
    main()