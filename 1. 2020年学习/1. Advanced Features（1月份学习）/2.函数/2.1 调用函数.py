#Python内置了很多有用的函数，我们可以直接调用
#调用abs函数：
print(abs(100))
print(abs(-300))

#abs函数只能接受一个参数，而max可以接受多个参数
print(max(1, 2, 23))

#数据类象转换
#Python内置的常用函数还包括数据类型转换函数，比如int()函数可以把其他数据类型转换为整数：
a = '123'
int(a)
print(a)

print(int(12.3))   #转换成整型

print(float('12')) #转换成浮点型

print(str(123))  #转换成字符串

print(bool(1))  #布尔类型

print(bool(''))

#函数名其实就是指向一个函数对象的引用，完全可以把函数名赋给一个变量，相当于给这个函数起了一个“别名”：
a = abs #变量名a指向abs
b = a(-1)
print(b)

#作业：请利用Python内置的hex()函数把一个整数转换成十六进制表示的字符串：
n1 = 255
n2 = 1000
print('n1 = %s \nn2 = %s' % (hex(n1), hex(n2)))

