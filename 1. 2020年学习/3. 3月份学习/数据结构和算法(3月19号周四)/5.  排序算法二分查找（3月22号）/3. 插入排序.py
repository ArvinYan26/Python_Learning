#插入排序
#最有复杂度O（n），
#最坏时间复杂度O（n^2）
#稳定性：稳定

def insert_sort(alist):
    """插入排序"""
    n = len(alist)
    #外层循环，从右边的无序序列中取出来多少元素执行此过程
    for j in range(1, n):
        #j = [1, 2, 3, ...., n-1]
        #j代表内层循环起始值
        i = j
        #执行从右边的无序序列中取出第一个元素，即i的位置的元素，然后插入到前边的正确位置中
        while i > 0:
            if alist[i] < alist[i-1]:
                alist[i], alist[i-1] = alist[i-1], alist[i]
                i -= 1
            else:
                break
if __name__ == "__main__":
    li = [54, 26, 93, 17, 77, 31, 44, 55, 20]
    print(li)
    insert_sort(li)
    print(li)