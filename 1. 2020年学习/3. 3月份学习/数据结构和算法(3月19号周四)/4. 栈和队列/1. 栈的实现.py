#用顺序表来写，list（链表），
#堆栈的操作，压栈（入栈）
#特点：后进先出

class Stack(object):
    """栈"""
    def __init__(self):
        self.__list = [] #__私有的，

    def push(self, item):
        """添加一个新的元素,压栈"""
        self.__list.append(item) #list（顺序表）尾部操作时时间复杂度是O（1），头部是O（n），所以用尾部添加


    def pop(self):
        """弹出栈顶元素，取出元素"""
        return self.__list.pop()

    def peek(self):
        """返回栈顶元素"""
        if self.__list:
            return self.__list[-1] #返回list最后一个元素，后进的是最后的
        else:
            return None

    def is_empty(self):
        """判断栈是否为空"""
        return self.__list == []

#0, [] , {}, (), 都是假，可以直接判断################################33

    def size(self):
        """返回栈的元素个数"""
        return len(self.__list)

if __name__ == "__main__":
    s = Stack()
    s.push(1)
    s.push(2)
    s.push(3)
    s.push(4)
    print(s.pop())
    print(s.pop())
    print(s.pop())
    print(s.pop())