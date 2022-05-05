def quick_sort(alist):
    """快速排序"""
    n = len(alist)
    mid_value = alist[0] #假设第一个是中间元素
    low = 0
    high = n-1 #表示最后一个元素索引

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
        #循环结束，说明不能移动了，那就交换high与low的值，high位置没有元素了，先不动high
        alist[high] = alist[low]

if __name__ == "__main__":
    li = [54, 26, 93, 17, 77, 31, 44, 55, 20]
    print(li)
    quick_sort(li)
    print(li)