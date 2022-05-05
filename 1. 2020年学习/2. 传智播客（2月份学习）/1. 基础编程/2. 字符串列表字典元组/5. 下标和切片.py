
name = "liumeizhe"
#从第四个字母开始取
print(name[3:])

print(name[2:-1:2]) #从左向右取，取不到倒数第一个，只能取到倒数第三个

print(name[::2])


#逆序
print(name[-1:0:-1])#逆序，但是取不到0这个元素，所以到第二个元素
print(name[-1::-1]) #取了全部
print(name[::-1])  #步长是-1， 所以是对字符串逆序
