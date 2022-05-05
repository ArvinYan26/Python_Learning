#插入排序的改进版本
#最优时间复杂度：根据步长序列的不同而不同
#最坏时间复杂度：O（n^2）
#稳定性：不稳定

def shell_sort(alist):
    """希尔排序"""
    n = len(alist)
    gap = n // 2
    #最外层控制gap步长
    while gap >= 1:
        for i in range(gap, n): #从指定位置开始选择要比对的元素
            while i > 0: #当某个要比较的元素一定到0位置就会停止，所以，条件是i>0
                if alist[i] < alist[i - gap]:
                    alist[i], alist[i - gap] = alist[i - gap], alist[i]
                    i -= gap  # 交换完以后，继续向前比对
                else:
                    break  # 对比一次以后，在进行第二次对比的时候发现不用交换直接退出
    #循环结束后，gap缩小一倍
        gap //= 2

if __name__ == "__main__":
    li = [54, 26, 93, 17, 77, 31, 44, 55, 20]
    print(li)
    shell_sort(li)
    print(li)