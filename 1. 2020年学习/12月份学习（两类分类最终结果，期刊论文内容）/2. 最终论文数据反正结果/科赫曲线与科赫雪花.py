
import turtle

#科赫曲线
turtle.pensize(4)
turtle.pencolor('green')
turtle.penup()
turtle.goto(-100,0)
turtle.pendown()

#抽象步骤，如果是0阶，只需前行；如果是一阶，需要前行，转向，前行，转向，前行，转向，前行，
#共有的是前行，阶数需要控制转向的次数，所以边界是0阶，只需前行
def keke_line(n=2,len=360):
    if n==0:
        turtle.fd(len)
    else:
        for i in [0,60,-120,60]:
            turtle.left(i)
            keke_line(n-1,len/3)

def kehe(len,n):
    if n == 0:
        turtle.fd(len)
    else:
        for i in [0,60,-120,60]:
            turtle.left(i)
            kehe(len / 3, n - 1)


lenth = 400
level = 3
du = 120
def main():
    turtle.penup()
    turtle.goto(-100,100)
    turtle.pensize(2)
    turtle.color('green')
    turtle.pendown()

    kehe(lenth,level)
    turtle.right(du)
    kehe(lenth, level)
    turtle.right(du)
    kehe(lenth, level)
    turtle.right(du)




    turtle.hideturtle()
    turtle.done()


if __name__ == '__main__':

    #main()

    keke_line()


    turtle.hideturtle()
    turtle.done()

