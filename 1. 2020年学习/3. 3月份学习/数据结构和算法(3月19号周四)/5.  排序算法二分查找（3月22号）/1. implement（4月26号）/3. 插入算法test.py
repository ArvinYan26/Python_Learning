#插入排序
#最有复杂度O（n），
#最坏时间复杂度O（n^2）
#稳定性：稳定
def insert_sort(alist):
    """插入排序"""
    n = len(alist)
    for j in range(1, n): #让j控制从第一个元素开始和其那边的元素对比
        i = j #让i=j, 即可
        while i > 0:
            if alist[i] < alist[i-1]:
                alist[i], alist[i-1] = alist[i-1], alist[i]
                i -= 1  #交换完以后，继续向前比对
            else:
                break  #对比一次以后，在进行第二次对比的时候发现不用交换直接退出


if __name__ == "__main__":
    li = [54, 26, 93, 17, 77, 31, 44, 55, 20]
    print(li)
    insert_sort(li)
    print(li)