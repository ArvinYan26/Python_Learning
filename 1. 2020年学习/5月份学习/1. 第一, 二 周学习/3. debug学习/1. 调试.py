"""
info reference : https://zhuanlan.zhihu.com/p/62610785
comment: for 循环中的下划线，在 Python 中是占位符的意思，因为单纯的循环两次而已，并不用到它的循环结果
"""

#for 循环中的下划线，在 Python 中是占位符的意思，因为单纯的循环两次而已，并不用到它的循环结果
def sum_demo(x, y):
    for _ in range(2):
        x += 1
        y += 1
        result = x +y
    return result

if __name__ == '__main__':
    result = sum_demo(1, 1)
    print(result)

#断点调试，英文 breakpoint。用大白话来解释下，断点调试其实就是在程序自动运行的过程中，你在代码某一处打上了断点，
#当程序跑到你设置的断点位置处，则会中断下来，此时你可以看到之前运行过的所有程序变量。
