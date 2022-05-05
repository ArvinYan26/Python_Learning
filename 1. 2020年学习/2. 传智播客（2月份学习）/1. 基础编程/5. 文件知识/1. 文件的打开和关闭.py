#举个例子,
f = open("test.py", "r")  #只读方式
f = open("test1.py", "w") #若test1存在，删除掉里面的内容，重新写，如果不存在，创建test1，开始写内容

f.close()  #关闭文件