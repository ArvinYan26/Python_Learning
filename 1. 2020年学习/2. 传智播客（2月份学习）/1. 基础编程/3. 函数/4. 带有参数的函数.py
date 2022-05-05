def sum_2_nums(a, b):
    #a = float(input("请输入a的值："))
    #b = float(input("请输入b的值："))
    result = a + b
    print("%d + %d = %d " %(a, b, result))

num1 = int(input("请输入第1个数字："))
num2 = int(input("请输入第2个数字："))

#调用函数
sum_2_nums(num1, num2)