#用sorted排序,
#sort()与sorted()的不同在于，sort是在原位重新排列列表，而sorted()是产生一个新的列表。
#sort 是应用在 list 上的方法，
#sorted 可以对所有可迭代的对象进行排序操作。

#列表全部是数字，就直接顺序排序,从小到大
nums = [11, 333, 4, 56, 23]
nums.sort()
print(nums)
#倒叙排序：从大到小
nums.sort(reverse=True)
print(nums)

#将所有元素顺序倒过来
nums.reverse()
print(nums)

"""廖雪峰内容
l = sorted(nums)
print(l)
#倒叙排序
l = sorted(nums, reverse=True)
print(l)
"""
#关键字lambda表示匿名函数，冒号前面的x表示函数参数。
#匿名函数有个限制，就是只能有一个表达式，不用写return，返回值就是该表达式的结果。
#用匿名函数有个好处，因为函数没有名字，不必担心函数名冲突。此外，匿名函数也是一个函数对象，也可以把匿名函数赋值给一个变量，再利用变量来调用该函数：

infors = [{"name":"laownag", "age": 10}, {"name":"laoli", "age": 12}, {"name":"laoyan", "age": 27}]
infors.sort(key=lambda x:x['name']) #把infors中的每一个元素（字典）的引用传给x，然后x按照name的排序
print(infors)

infors.sort(key=lambda x:x['age']) #按照年龄排序
print(infors)