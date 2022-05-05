#列表增删改查都行，元组不能改（相当于只读文件）

#Python内置的一种数据类型是列表：list。list是一种有序的集合，可以随时添加和删除其中的元素。
#比如，列出班里所有同学的名字，就可以用一个list表示：
classmates = ['A', 'B', 'C', 'D']
print(classmates)
#变量classmates就是一个list。用len()函数可以获得list元素的个数：
print(len(classmates))

#用索引访问list的元素
print(classmates[0])

classmates.append('Adma')  #添加到列表结尾，
classmates.insert(2, 'F')  #添加到索引为2的位置
print(classmates)

classmates.pop()    #删除列表末尾的元素
print(classmates)

classmates.pop(2)  #删除索引为2的列表元素

#要把某个元素替换成别的元素，可以直接赋值给对应的索引位置：
classmates[1] = 'Alice'
print(classmates)

#list元素类型不一样
L = ['Alice', 123, True]
#list元素也可以是另一个list
s = ['python', 'java', ['asp', 'pha'], 'scheme']
print(len(s))
print(s)
print(s[2])

#元组tuple
#另一种有序列表叫元组：tuple。tuple和list非常类似，但是tuple一旦初始化就不能修改，比如同样是列出同学的名字：
classmates = ('A', 'B', 'C', 'D')

#不可变的tuple有什么意义？因为tuple不可变，所以代码更安全。如果可能，能用tuple代替list就尽量用tuple。
#tuple的陷阱：当你定义一个tuple时，在定义的时候，tuple的元素就必须被确定下来，比如：
t = (1, 2)
print(t)

#定义一个空的tuple
t = ()
print(t)

#定义一个只有1个元素的tuple，如果你这么定义：
t = (1)
print(t)

#定义的不是tuple，是1这个数！这是因为括号()既可以表示tuple，又可以表示数学公式中的小括号，这就产生了歧义，因此，
#Python规定，这种情况下，按小括号进行计算，计算结果自然是1。
#所以，只有1个元素的tuple定义时必须加一个逗号,，来消除歧义：
t = (1,)
print(t)

#Python在显示只有1个元素的tuple时，也会加一个逗号,，以免你误解成数学计算意义上的括号。
#最后来看一个“可变的”tuple：
t = ('a', 'b', ['A', 'B'])
print(t)
t[2][0] = 'Y'
t[2][1] = 'X'
print(t)



