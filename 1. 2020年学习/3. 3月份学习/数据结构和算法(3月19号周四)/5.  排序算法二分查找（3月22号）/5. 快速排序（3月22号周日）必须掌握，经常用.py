#快速排序,核心方法递归嵌套（经常使用，很重要）
#最优复杂度：O（n*logn）
#最坏时间复杂度：O（n^2）
#稳定性：不稳定

def quick_sort(alist, first, last):
    """快速排序"""
    if first >= last:
        return  #如果只有一个元素，那就直接返回，不进行任何操作
    mid_value = alist[first]
    low = first  #新序列的起始位置
    high = last  #新序列的终止位置
    while low < high:
        #让high游标左移
        while low < high and alist[high] >= mid_value:
            #>= :是要把相等的值全部放在右边去
            high -= 1
        alist[low] = alist[high]

        while low < high and alist[low] < mid_value:
            low += 1
            alist[high] = alist[low]

    #从循环退出时， low = high
    alist[low] = mid_value

    #对low左边的列表进行快速排序
    quick_sort(alist, first, low-1)
    #对列表右边序列进行排序
    quick_sort(alist, low+1, last)

if __name__ == "__main__":
    li = [54, 26, 93, 17, 77, 31, 44, 55, 20]
    print(li)
    quick_sort(li, 0, len(li)-1) #将0传给first，len(li)-1 传给last
    print(li)


