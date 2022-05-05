#sorted:排序算法
#排序也是在程序中经常用到的算法。无论使用冒泡排序还是快速排序，排序的核心是比较两个元素的大小。如果是数字，我们可以直接比较，但如果是字符串或者两个dict呢？直接比较数学上的大小是没有意义的，因此，比较的过程必须通过函数抽象出来。
#Python内置的sorted()函数就可以对list进行排序：
L = sorted([36, 5, -12, 9, -21])
print(L)

#此外，sorted()函数也是一个高阶函数，它还可以接收一个key函数来实现自定义的排序，例如按绝对值大小排序：
L = sorted([36, 5, -12, 9, -21], key=abs)
print(L)

s = ['bob', 'about', 'Zoo', 'Credit']
L = sorted(s, key=str.lower) #先把所有字母都变为小写的再排序
print(L)

s = ['bob', 'about', 'Zoo', 'Credit']
L = sorted(s, key=str.lower, reverse=True)  #reverse：逆向的，反向的
print(L)

L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]
def by_name(t):
    t = t[0]
    return t
L2 = sorted(L, key=by_name)
print(L2)

#按成绩排序
L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]

def by_score(t):
    t=t[1]
    return t

L2 = sorted(L, key=by_score)
print(L2)