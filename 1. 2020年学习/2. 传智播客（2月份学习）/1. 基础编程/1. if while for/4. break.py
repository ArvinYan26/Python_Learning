"""
#打印100以内的数
i = 1
while i<=100:
    if i%2==0:
        print(i)
    i+=1

i = 1
while i<=5:
    print("----------------")
    if i==3:
        break  #相当于此时停止，结束while，紧接着执行while下面的代码

    print(i)
    i+=1
"""
#打印20个偶数，停止while循环
i = 1
num = 0
while i <= 100:

    if i%2==0:
        #print(i)
        num+=1
    if num==20:
        break
    i += 1
"""
i = 1
num = 0
while i <= 10:
    i+=1
    print("----------")
    if i==3:
        continue  #当i==3的时候不执行，直接跳到初始继续下一次的执行，相当于跳过执行

    print(i)
"""
print(num)