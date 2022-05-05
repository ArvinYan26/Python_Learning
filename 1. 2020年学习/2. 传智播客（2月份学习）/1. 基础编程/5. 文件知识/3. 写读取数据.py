f = open("xxx.txt", "w")

f.write("hahaha")

f.close()

#读取数据
f = open("xxx.txt", "r")

content = f.read()
print(content)

f.close()