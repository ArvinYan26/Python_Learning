#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
#语法：enumerate(sequence, [start=0])
#sequence -- 一个序列、迭代器或其他支持迭代对象。
#start -- 下标起始位置。

#for 循环一般案例
i = 0
seq = ["one", "two", "three"]
for element in seq:
    print(i, seq[i])
    i += 1
#for循环使用enumerate
seq = ["one", "two", "three"]
for i, element in enumerate(seq):
    print(i, element)
"""
0 one
1 two
2 three
"""
