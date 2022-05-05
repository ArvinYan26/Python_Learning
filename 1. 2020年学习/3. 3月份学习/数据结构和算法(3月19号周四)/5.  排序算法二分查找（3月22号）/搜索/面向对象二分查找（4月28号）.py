#只能作用于列表
#操作对象必须是经过排序后的
#对象必须是有序的
#最优时间复杂度：O(1)
#最坏时间复杂度：O(nlogn)

#递归版本查找
#一般遇到有递归函数的程序时，一般不用类写，那样反而会增让事情变得更加复杂，传参数很麻烦

class BinarySearch(object):

    def __init__(self, list, elem):
        self.alist = list
        self.n = len(list)
        self.item = elem

    def binary_search(self, alist):
        """递归版本，二分查找"""
        #n = len(self.alist)
        if self.n > 0:
            mid = self.n // 2
            if self.alist[mid] == self.item:
                return True
            elif self.item < self.alist[mid]:
                return self.binary_search(self.alist[:mid])
            else:
                return self.binary_search(self.alist[mid+1:])
        return False #循环结束也没有找到

    def binary_search_2(self):
        """非递归版本"""
        n = len(self.alist)
        first = 0
        last = n-1
        while first < last:
            mid = (first+last)//2
            if self.item == self.alist[mid]:
                return True
            elif self.item < self.alist[mid]:
                last = mid-1
            else:
                first = mid+1
        #循环结束也没找到，就返回False
        return False



def main():
    li = [1, 11, 19, 22, 33, 33, 43, 74]
    bin = BinarySearch(li, 11)
    result =bin.binary_search()
    print(result)
    result =bin.binary_search()
    print(result)

    #非递归版本查找
    result =bin.binary_search_2()
    print(result)
    result =bin.binary_search_2()
    print(result)

if __name__ == '__main__':
    main()

