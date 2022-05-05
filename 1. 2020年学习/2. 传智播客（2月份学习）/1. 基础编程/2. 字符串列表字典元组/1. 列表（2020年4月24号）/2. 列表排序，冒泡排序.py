#产生10个1-50之间的随机数，添加到列表中
import random

#产生10个不同的随机数
#如果相同就重新产生，如果用for循环，个数不一定是10十个，所以用while循环
random_list = []
i = 0
while i < 10:
    rand = random.randint(1, 20)
    if rand not in random_list:
        random_list.append(rand)
        i += 1
print(random_list)




