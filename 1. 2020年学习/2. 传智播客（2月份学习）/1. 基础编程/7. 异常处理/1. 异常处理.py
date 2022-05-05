#名字命名异常处理
try:
    print(num)
    print("-----1-------")

except NameError:
    print("捕捉到异常，正在处理----")  #出现此类异常的处理方式

print("-------2------")

#文件不存在处理
try:
    open("xxx.txt")
    print("-----1-------")

except FileNotFoundError:
    print("文件不存在----")  #出现此类异常的处理方式

print("-------2------")

#两个合并，出现异常就会挨着判断属于哪种异常，然后采取相应措施
try:
    open("xxx.txt")
    print(num)
    print("-----1-------")

except NameError:
    print("捕捉到异常，正在处理----")  #出现此类异常的处理方式

except FileNotFoundError:
    print("文件不存在----")  #出现此类异常的处理方式

print("-------2------")

#所有的异常处理方式一样
try:
    print(num)
    open("xxx,txt")
    print("-----1-------")

except (NameError, FileNotFoundError):
    print("捕捉到异常，正在处理----")  #出现此类异常的处理方式

print("-------2------")

#能捕捉到所有异常的except Exception：
try:
    11/0
    print(num)
    open("xxx,txt")
    print("-----1-------")

except (NameError, FileNotFoundError): #捕获指定异常
    print("捕捉到异常，正在处理----")  #出现此类异常的处理方式

except Exception as ret: #捕获所有异常,且将错误原因存放在ret中，ret只是一个名字，可以随便写
    print("如果用了Exception，那意味着只要上面的except没有捕捉到异常常，这个except一定能捕捉到")
    print(ret)
else:
    print("没有异常才能执行的功能")
print("-------2------")

#没有异常执行的功能else
try:
    """
    11/0
    print(num)
    open("xxx,txt")
    print("-----1-------")
    """
    print("-----1-------") #这个不是异常

except (NameError, FileNotFoundError): #捕获指定异常
    print("捕捉到异常，正在处理----")  #出现此类异常的处理方式

except Exception as ret: #捕获所有异常,且将错误原因存放在ret中，ret只是一个名字，可以随便写
    print("如果用了Exception，那意味着只要上面的except没有捕捉到异常常，这个except一定能捕捉到")
    print(ret)
else:
    print("没有异常才能执行的功能")
print("-------2------")

#finally
try:
    11/0
    print(num)
    open("xxx,txt")
    print("-----1-------")

except (NameError, FileNotFoundError): #捕获指定异常
    print("捕捉到异常，正在处理----")  #出现此类异常的处理方式

except Exception as ret: #捕获所有异常,且将错误原因存放在ret中，ret只是一个名字，可以随便写
    print("如果用了Exception，那意味着只要上面的except没有捕捉到异常常，这个except一定能捕捉到")
    print(ret)
else:
    print("没有异常才能执行的功能")

finally: #不管是否有异常都执行（例如读取文件，不管有异常没，最后都得关闭文件）
    print("--------finally-------")

print("-------2------")

