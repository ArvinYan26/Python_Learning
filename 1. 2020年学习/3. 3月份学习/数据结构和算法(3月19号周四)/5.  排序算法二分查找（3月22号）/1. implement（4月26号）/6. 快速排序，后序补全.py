#最优时间复杂度：O(nlogn), 正常情况下
#最坏时间复杂度：O(n2)， 某一边（左边或者右边），每一次只能分出来一个元素
#稳定性：不稳定

def quick_sort(alist, first, last):
    """
    快速排序
            快速排序的介绍
    快速排序(quick sort)的采用了分治的策略
    分治策略指的是：
    将原问题分解为若干个规模更小但结构与原问题相似的子问题。递归地解这些子问题，然后将这些子问题的解组合为原问题的解。

    快排的基本思想是：
    在序列中找一个划分值，通过一趟排序将未排序的序列排序成 独立的两个部分，其中左边部分序列都比划分值小，右边部分的序列比划分值大，
    此时划分值的位置已确认，然后再对这两个序列按照同样的方法进行排序，从而达到整个序列都有序的目的。
    :param :alist:传入的列表， first:列表起始所以， last：列表最后元素索引
    """
    if first >= last:
        return
    mid_value = alist[first] #假设第一个是中间元素
    low = first
    high = last #表示最后一个元素索引

    while low < high:
        #下面两个循环交替执行，循环一个来回以后，还要继续，所以需要外层循环控制
        #high左移
        while low < high and alist[high] >= mid_value:
            high -= 1
        #循环结束，说明不能移动了，那就交换high与low的值，low位置没有元素了，先不动low
        alist[low] = alist[high]
        #low右移
        while low < high and alist[low] < mid_value:
            low += 1
        #循环结束，说明不能移动了，那就交换low和mid_value的值，high位置没有元素了，先不动high
        alist[high] = alist[low]

    #从循环退出时，low==high， 将中间值放在low的位置（high的位置也一样，因为此时low=high）
    alist[low] = mid_value

    #函数调用函数本身，递归思想
    #对low左边的列表进行快速排序
    quick_sort(alist, first, low-1)

    #对low右边的列表进行快速排序
    quick_sort(alist, low+1, last)




if __name__ == "__main__":
    li = [54, 26, 93, 17, 77, 31, 44, 55, 20]
    print(li)
    quick_sort(li, 0, len(li)-1)
    print(li)