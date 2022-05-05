""""
归并排序
#拆分，再组合
最优时间复杂度：O(nlogn)
最坏时间复杂度：O(nlogn)
稳定性：稳定
"""
def merge_sort(alist):
    """归并排序"""
    n = len(alist)
    #拆分列表
    if n <= 1: #当到最后只有一个元素的时候，直接返回列表就行，因为也需要接收
        return alist
    mid = n // 2
    #调用函数本身完成相同步骤的拆分，递归思想
    #left是采用归并排序后的新的有序的列表
    left_li = merge_sort(alist[:mid]) #此处时传入参数是切片后形成的新的列表

    # left是采用归并排序后的新的有序的列表
    right_li = merge_sort(alist[mid:])

    #将两个有序的序列合并成整个序列
    #merge（left, right）， 合并需要有两个游标协助
    left_point, right_point = 0, 0 #游标都是从零开始
    result = []  #用于存放合并的有序的新列表，不是原来的那个，
    #任何一个游标走到头到代表，循环结束,所以游标没有走到头的时候执行循环
    while left_point < len(left_li) and right_point < len(right_li):
        if left_li[left_point] < right_li[right_point]:
            result.append(left_li[left_point])
            left_point += 1
        else:
            result.append(right_li[right_point])
            right_point += 1
    #循环结束后，把左右剩下的元素添加到新列表result中
    result += left_li[left_point:]
    result += right_li[right_point:]
    return result #因为需要将新列表存取，递归嵌套后需要接收这个返回值

def main():
    """主函数，执行框架"""
    li = [54, 26, 93, 17, 77, 31, 44, 55, 20]
    print(li)
    # sorted_li 用来接收result的返回值
    sorted_li = merge_sort(li)  # 将0传给first，len(li)-1 传给last
    print(li)
    print(sorted_li)


if __name__ == "__main__":
    main()
