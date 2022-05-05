#选择排序,
#时间复杂度O（n^2）,
#稳定性：不稳定
#li = [54, 26, 93, 17, 77, 31, 44, 55, 20]
def select_sort(alist):
    """选择排序,时间复杂度O（n^2）"""
    n = len(alist)
    for j in range(n-1): #j范围：0~n-2
        min_index = j
        for i in range(j+1, n):
            if alist[min_index] > alist[i]:
                min_index = i
        alist[j], alist[min_index] = alist[min_index], alist[j]

if __name__ == "__main__":
    li = [54, 26, 93, 17, 77, 31, 44, 55, 20]
    print(li)
    select_sort(li)
    print(li)
