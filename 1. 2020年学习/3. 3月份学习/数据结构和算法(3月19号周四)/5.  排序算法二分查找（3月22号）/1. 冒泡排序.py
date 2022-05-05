#冒泡排序稳定性：稳定

def bubble_sort(alist):
    """冒泡排序"""
    n = len(alist)
    for j in range(n-1): #控制冒泡排序的遍数，要走几遍
        count = 0
        for i in range(0, n-1-j): #控制一遍冒泡，需要冒几次才能从头走到位
            #班长从头头走到尾排好一次，下次就少一次，所以下表长度是动态变化的
            if alist[i] > alist[i + 1]:
                alist[i], alist[i + 1] = alist[i + 1], alist[i]
                count += 1
        if 0 == count:
            return
if __name__ == "__main__":
    li = [54, 26, 93, 17, 77, 31, 44, 55, 20]
    print(li)
    bubble_sort(li)
    print(li)