#如果要访问循环体内每个元素的索引，请使用内置的 enumerate 函数：

animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# Prints "#1: cat", "#2: dog", "#3: monkey", each on its own line

#列表推导式
#普通列表循环
nums = [0, 1, 2, 3, 4]
l = []
for x in nums:
    l.append(x**2)
print(l)
#列表推导式
nums = [0, 1, 2, 3, 4]
l = [x**2 for x in nums]
print(l)

#添加条件
nums = [0, 1, 2, 3, 4]
l = [x**2 for x in nums if x%2 ==0]
print(l)

