#插入排序的改进版本
#最优时间复杂度：根据步长序列的不同而不同
#最坏时间复杂度：O（n^2）
#稳定性：不稳定

def shell_sort(alist):
    """希尔排序"""
    n = len(alist)
    #gap=4,(比如说从4个间隔开始)
    gap = n // 2  #取整除部分，9//2，取4

    #gapb变化到0之前，插入算法执行的次数
    while gap > 0:
    # 插入算法，与普通插入算法的不同就是gap步长
        #外层循环j,控制需要循环的所有元素
        for j in range(gap, n):
            #j= [gap, gap+1, ...., n-1]
            #内层循环，控制交换,
            i = j
            while i > 0:  #直到最后元素挪到最前边，即i=0时停止
                if alist[i] < alist[i-gap]:
                    alist[i], alist[i-gap] = alist[i-gap], alist[i]
                    i -= gap
                else:
                    break
        #缩短gap步长,最短可以是1
        gap //= 2



if __name__ == "__main__":
    li = [54, 26, 93, 17, 77, 31, 44, 55, 20]
    print(li)
    shell_sort(li)
    print(li)
