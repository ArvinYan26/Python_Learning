class Sort_algorthim(object):
    def __init__(self, alist):
        self.list = alist
        self.n = len(alist)
        self.gap = self.n // 2
        self.count = 0

    def quick_sort(self, first, last):
        """快速排序"""
        #当左右两边的游标没有错过就继续执行，否则返回，不执行
        if first >= last:
            return
        mid_value = self.list[first] #假第一个元素时中间值
        low = first
        high = last  #表示最后元素的索引

        while low < high:
            while low < high and self.list[high] >= mid_value:
                high -= 1
                #条件不符合，说明self.list[high] < mid_value, 就交换值
            self.list[low] = self.list[high]
            while low < high and self.list[low] < mid_value:
                low += 1
            self.list[high] = self.list[low]

        #循环退出以后，将low位置的值变为mid_value的值
        self.list[low] = mid_value

        #重复此过程，函数调用函数本身，递归思想
        #对low左边的序列进行快排
        #此处的self.list已经变成了新的切片后的子序列
        Sort_algorthim.quick_sort(self.list, first, low-1)

        #对右low有右边的序列进行快排
        Sort_algorthim.quick_sort(self.list, low+1, last)


def main():
    l = [1, 19, 74, 22, 33, 11, 43, 33]
    print("未排序的列表：", end="")
    print(l)
    sort = Sort_algorthim(l)
    sort.quick_sort(0, len(l)-1)
    print("快速排序后的列表：", end="")
    print(l)


if __name__ == '__main__':
    main()

