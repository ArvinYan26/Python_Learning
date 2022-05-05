




def sort_box(pre_box):

    results = []
    li = []
    for i in range(len(pre_box)):
        li.append(pre_box[i][4])

    li_2 = set(li)
    if len(li_2) == len(pre_box):
        pre_box.sort(key=lambda det: (det[0]))  # 按需要的数据索引排序
    else:
        # 思路说明：1 先按照得分排名排序
        #         2 再找到新的数组中的得分相同的框按照宽度排序调整顺序
        pre_box.sort(key=lambda det: (det[0]))  # 按需要的数据索引排序
        # for i in range(len(pre_box)):
        #     for j in range(i+1, len(pre_box)):
        #         if pre_box[i][4] == pre_box[j][4]:
        #             results.append()





