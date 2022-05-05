#最后看一个有问题的条件判断。很多同学会用input()读取用户的输入，这样可以自己输入，程序运行得更有意思：
s = input('birth:')
birth = int(s)
if birth < 2000:
    print('oo前')
else:
    print('00后')


#小明身高1.75，体重80.5kg。请根据BMI公式（体重除以身高的平方）帮小明计算他的BMI指数，并根据BMI指数：
#低于18.5：过轻
#18.5-25：正常
#25-28：过重
#28-32：肥胖
#高于32：严重肥胖
#用if-elif判断并打印结果：

name = input('请输入您的姓名： ')
age = float(input('请输入您的年龄（岁):'))
height = float(input('请输入您的身高：'))
weight = float(input('请输入您的体重：'))
bmi = weight / (height**2)
if bmi < 18.5 :
    print('您的BMI为：%.1f, 多吃点！' % bmi)
elif 18.5 <= bmi <= 25:
    print('您的BMI为：%.1f, 继续保持！' % bmi)
elif 25 < bmi <= 28:

    print("您的BMI为:%.1f,有点偏重了噢，控制下自己吧！" % bmi)

elif 28 < bmi <= 32:

    print("您的BMI为:%.1f,您太胖了，快行动起来减肥吧！" % bmi)

elif bmi > 32:

    print("您的BMI为:%.1f,您到底吃了什么，自己想想吧！" % bmi)

else :

    print("您是外星人！")