"""
f = open("test.py", "r") #如果不存在，崩溃
f.read()   #把整个文件读取出来
f.read(1)   #挨着一个一个字节的读取文件内容
f.read(2)   #挨着两个两个字节的读取文件内容
"""
f = open("test1.py", "w")
f1 = f.write("haha~")
f2 = f.write("\nhaha~")  #\n:表示一个字节
print(f1)  #5:表示5个字
print(f2)

f.close()
