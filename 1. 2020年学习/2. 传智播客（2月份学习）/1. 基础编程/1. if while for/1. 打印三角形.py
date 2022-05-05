#打印矩形
i = 1
while i <= 5:
    j = 1
    while j<=5:
        print("*", end="  ")
        j += 1
    print(" ")
    i += 1
print("-"*100)

#打印三角形
i = 1
while i <= 5:
    j = 1
    while j<=i:
        print("*", end="  ")
        j += 1
    print("")  #换行符
    i += 1
