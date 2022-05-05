#只能作用于列表
#操作对象必须是经过排序后的
#对象必须是有序的,这点很重要
#最优时间复杂度：O(1)
#最坏时间复杂度：O(nlogn)

#递归版本查找
def binary_search(alist, item):
    """二分法查找"""
    n = len(alist)
    if n > 0:
        mid = n//2
        if alist[mid] == item:
            return True  #找到了就返回True
        elif item < alist[mid]:
            return binary_search(alist[:mid], item)
        else:
            return binary_search(alist[mid+1:], item)
    return False  #循环结束没又找到，就返回Flase


#非递归版本二分法查找
def binary_search_2(alist, item):
    """"非递归版本"""
    n = len(alist)
    first = 0
    last = n-1
    while first <= last:
        mid = (first+last)//2
        if alist[mid] == item:
            return True
        elif item < alist[mid]:
            last = mid-1
        else:
            first = mid+1
    return False


if __name__ == "__main__":
    li = [1, 11, 19, 22, 33, 33, 43, 74]
    print(binary_search(li, 19))
    print(binary_search(li, 100))
    print(binary_search_2(li, 19))
    print(binary_search_2(li, 100))

