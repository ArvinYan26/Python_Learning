#在Python中，定义一个函数要使用def语句，依次写出函数名、括号、括号中的参数和冒号:，然后，在缩进块中编写函数体，函数的返回值用return语句返回。
#返回多个值
import math
def move (x, y , step, angle = 0):
    nx = x + step * math.cos(angle)
    ny = y - step * math.sin(angle)
    return nx, ny
x, y = move(100, 100, 60, math.pi / 6)
print(x, y)

#请定义一个函数quadratic(a, b, c)，接收3个参数，返回一元二次方程 ax^2+bx+c=0ax

import math
def quadratic(a, b, c):
    s = float(b**2-4*a*c)
    while a != 0:  # != :表示不等于的意思
        if s > 0:
            x1 = (-b + math.sqrt(s)/(2*a))
            x2 = (-b - math.sqrt(s)/(2*a))
            return x1, x2
        elif s == 0:
            x = (-b)/(2*a)
            return x
        else :
            return "无实数解"
a = float(input("请输入a的值："))
b = float(input("请输入b的值："))
c = float(input("请输入c的值："))
print('解为:', quadratic(a, b, c))