i = 1
while i <= 9:
    j = 1
    while j<=i:
        print("%d*%d=%d\t  "%(j, i, j*i), end="  ")  #\t:加table键，作用是对其齐
        j += 1
    print("")  #换行符
    i += 1

#计算1-100的和
sum = 0
for i in range(100):
    sum = sum + i
print(sum)