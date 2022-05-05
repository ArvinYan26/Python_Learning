#获取用户要复制的文件名
old_file_name = input("请输入要复制的文件名：")

#打开要赋值的文件
old_file = open(old_file_name, "r")

position = old_file_name.rfind(".")
new_file_name = old_file_name[:position] + "[复件]" + old_file_name[position:]

#新建一个文件
new_file = open("new_file_name", "w")

#从旧文件中读取数据，并且写入到新文件中去
contnet = old_file.read()
new_file.write(contnet)

#关闭两个文件
old_file.close()
new_file.close()