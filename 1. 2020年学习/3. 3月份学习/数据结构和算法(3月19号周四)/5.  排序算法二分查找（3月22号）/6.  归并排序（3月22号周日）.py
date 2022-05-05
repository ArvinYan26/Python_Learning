#归并排序，先分后合,与快速排序不同的是，拆分成新序列
#时间复杂度：O（n*logn）
#最坏时间复杂度：O（n*logn），但是额外的空间，用空间换时间
#稳定性：稳定
def merge_sort(alist):
    """归并排序"""
    n = len(alist)
    if n <= 1:
        return alist  #当最后拆分的序列只有一个元素的话，那么只需要将原序列返回即可
    mid = n // 2

    #left 采用归并排序后形成的有序的新列表
    left_li = merge_sort(alist[:mid])

    #right 采用归并排序后形成的有序的新列表
    right_li = merge_sort(alist[mid:])

    #将两个有序的子序列合成一个新的有序的序列
    #merge(left, right)
    left_pointer, right_pointer = 0, 0 #拆分的两个新序列的游标位置都从0开始
    result = []

    while left_pointer < len(left_li) and right_pointer < len(right_li):
        if left_li[left_pointer] < right_li[right_pointer]:
            result.append(left_li[left_pointer])
            left_pointer += 1
        else:
            result.append(right_li[right_pointer])
            right_pointer += 1

    result += left_li[left_pointer:]
    result += right_li[right_pointer:]
    return result  #需要返回这个函数，获取新序列

if __name__ == "__main__":
    li = [54, 26, 93, 17, 77, 31, 44, 55, 20]
    print(li)
    #sorted_li 用来接收result的返回值
    sorted_li = merge_sort(li) #将0传给first，len(li)-1 传给last
    print(li)
    print(sorted_li)