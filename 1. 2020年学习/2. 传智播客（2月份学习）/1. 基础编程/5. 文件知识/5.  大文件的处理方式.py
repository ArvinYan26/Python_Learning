f.readline() #每一行读取一次

f.readlines() #所有内容读取出来，并且用列表存起来

#读取大的文件，不用read，因为文件可能非常大，内容很多，需要用一个程序一次读一点，处理完了，再继续读取

#获取用户要复制的文件名
old_file_name = input("请输入要复制的文件名：")

#打开要赋值的文件
old_file = open(old_file_name, "r")

position = old_file_name.rfind(".")
new_file_name = old_file_name[:position] + "[复件]" + old_file_name[position:]

#新建一个文件
new_file = open("new_file_name", "w")

#从旧文件中读取数据，并且写入到新文件中去
##读取大的文件，不用read，因为文件可能非常大，内容很多，需要用一个程序一次读一点，处理完了，再继续读取
while True:
    contnet = old_file.read(1024)
    if len(contnet)==0:
        break
    new_file.write(contnet)

#关闭两个文件
old_file.close()
new_file.close()