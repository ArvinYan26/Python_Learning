#计算一个数的阶乘
"""
#以前知识
i = 1
result = 1
while i <= 4:
    result = result*i
    i += 1
print(result)
"""

#新的知识,递归：函数调用函数自己，而不是嵌套调用别的函数
def getNums(num):
    if num > 1:
        return num*getNums(num-1)
    else:
        return num
result = getNums(4)
print(result)

#注意：不能写死循环的递归，这样会造成内存溢出，闪退，程序死掉
def test():
    print("haha")
    test()

test()