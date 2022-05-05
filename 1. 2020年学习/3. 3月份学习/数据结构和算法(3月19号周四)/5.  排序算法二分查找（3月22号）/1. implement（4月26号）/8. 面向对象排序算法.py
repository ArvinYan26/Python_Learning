class Sort_algorthim(object):
    def __init__(self, alist):
        self.list = alist
        self.n = len(alist)
        self.gap = self.n // 2
        self.count = 0

    def bubble_sort(self):
        """
        冒泡排序
        从头开始，依次比较两个元素（内循环），一次循环节结束，最大的元素放在，这样的步骤持续n-1次，这是外层循环

        """
        #控制这样的循环要走多少次，循环的时候不要盯着范围看，只思考需要走多少次，范围就写多大，不要向下标的事情。
        for j in range(self.n-1): # 从0开始，循环n-1次
            #班长从头走到尾需要循环多少次，班长从头开始走，走到-1这个元素就结束，左移是n-1
            for i in range(self.n-1-j):
                if self.list[i] > self.list[i+1]:
                    self.list[i], self.list[i+1] = self.list[i+1], self.list[i]
                    #self.count += 1
            #if 0 == self.count: #如果self.count > 0, 说明进行交换，如果此时==0， 说明没践行交换，已经走到头了
                #return

    def select_sort(self): #
        """
        选择排序
        找未排序的最小元素，插入到前边已经续序列的合适位置
        假设第j个元素最小，然后从第j+1个元素到底n个元素中找最小的值，和第j个元素比较交换，循环了n-1-j次
        j是从0开始，直到第n-1个元素位置，循环了n-1次
        """
        for j in range(self.n-1):#到倒数第0个元素，索引为n-1
            min_index = j #假设从j开始，所以要从j+1位置开始找后边最小元素，最开始是从0开始
            for i in range(j+1, self.n): #从j+1元素开始找，到最后一个元素。循环n-j-1次
                if self.list[min_index] > self.list[i]:
                    min_index = i
            #循环结束以后找到了最小的元素，交换元素
            self.list[min_index], self.list[j] = self.list[j], self.list[min_index]

    def insert_sort(self):
        """
        插入排序，将未排序的序列插入到已排好的序列当中的适当位置
        :return:
        """
        for j in range(1, self.n): #从0开始，到最后，循环了n-1次
            i = j
            while i > 0:
                if self.list[i] < self.list[i-1]:
                    self.list[i], self.list[i-1] = self.list[i-1], self.list[i]
                    i -= 1
                else:
                    break #如果不需要交换，直接终止循环即可

    def shell_sort(self):
        """
        希尔排序
        相当于插入排序，只不过是步长变为了gap，不断缩短步长来减少计算量
        :return:
        """
        #只要self.gap >= 1, 就继续执行，
        while self.gap >= 1:
            for i in range(self.gap, self.n):
                while i > 0:
                    if self.list[i] < self.list[i-self.gap]:
                        self.list[i], self.list[i-self.gap] = self.list[i-self.gap], self.list[i]

                    else:
                        break
            self.gap //= 2

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
        self.quick_sort(first, low-1)

        #对右low有右边的序列进行快排
        self.quick_sort(low+1, last)

def merge_sort(alist):
    """归并排序"""
    n = len(alist)
    if n <= 1:
        return alist  #如果最后拆分的数据列表只有一个元素，那就直接返回这个列表

    mid = n // 2
    #从中间开始拆分序列为两部分
    left_l = merge_sort(alist[:mid])
    right_l = merge_sort(alist[mid:])

    #将两个子序列合并成分新序列，每个序列都需要初始位置的指针
    left_point, right_point = 0, 0
    result = [] #存储合并的新元素

    while left_point < len(left_l) and right_point < len(right_l):
        if left_l[left_point] < right_l[right_point]:
            result.append(left_l[left_point])
            left_point += 1
        else:
            result.append(right_l[right_point])
            right_point += 1
    #把左右列表中未排序的剩余的元素追加到排序后的列表
    result += left_l[left_point:]
    result += right_l[right_point:]
    return result



def main():
    l = [1, 19, 74, 22, 33, 11, 43, 33]
    print("未排序的列表：", end="")
    print(l)
    sort = Sort_algorthim(l)
    sort.bubble_sort()
    print("冒泡排序后的列表：", end="")
    print(l)

    sort = Sort_algorthim(l)
    sort.select_sort()
    print("选择排序后的列表：", end="")
    print(l)

    sort = Sort_algorthim(l)
    sort.insert_sort()
    print("插入排序后的列表：", end="")
    print(l)

    sort = Sort_algorthim(l)
    sort.shell_sort()
    print("希尔排序后的列表：", end="")
    print(l)

    sort = Sort_algorthim(l)
    sort.quick_sort(0, len(l)-1)
    print("快速排序后的列表：", end="")
    print(l)

    sorted_l = merge_sort(l)
    print("快速排序后的列表：", end="")
    print(l)




if __name__ == '__main__':
    main()

