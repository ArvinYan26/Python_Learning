#查找字符串
name = "liu mei zhe"
print(name.find("liu"))  #从左边开始找

print(name.rfind("mei"))  #从右边开始找


print(name.index("mei"))   #和find一样，不同是，index找不到返回错误，
                        #而find找不到就返回-1

#count统计个数，
print(name.count("i"))  #两个i字母，统计字符串中i的个数

#修改replace
new = name.replace("mei", "jin")
print(new)

#split:切割，将原有字符串切割成一个列表，然后可以运用for新欢进行下一步操作
snew = new.split(" ")
print(snew)

"""
#capitalize:将字符串的第一个字母大写
#title：将字符串的每一个单词的手字母大写
print(new.capitalize())
print(new.title())

#startswith 和 endswith :判断是否以什么开头和以什么结尾
#加入有很多文件，查找txt结尾的
file_name = "xxx.txt"
print(file_name.endswith(".txt"))

#startswith
print(name.startswith("liu"))

#lower:把所有字母变为小写，upper：把所有字母变为大写

exit_flag = "Yes"
print(exit_flag.lower())
print(exit_flag.upper())

#rjust，center和lstrip
#歌词居中显示
lyric = "你就是我的小星星，我也是你的小星星"
print(lyric.center(50))  #一行的宽度
#print(lyric.ljust(50))  #靠左排列，宽度50
#print(lyric.rjust(50))   #靠右排列，宽度50

#strip ，lstrip, rstrip  ,祛除空格，数据爬取，清洗时用到，很重要
test = lyric.center(50)
print(test.lstrip()) #去掉左边的空格
print(test.rstrip())
print(test.strip())

#partition:从左边查找第一个所定字符，然后切割，结果不会丢失所定字符串，返回的是一个元组（只找一个关键字符串）
#而split返回的是找到所有所定字符进行切割，返回一个列表，且所定字符串切完以后丢弃，对比结果如下
l = test.partition("星")
print(l)
#rpartition:是从右边开始找第一个所定字符串
x = test.rpartition("星")
print(x)
#split:切割，将原有字符串切割成一个列表，然后可以运用for新欢进行下一步操作
L = test.split("星")
print(L)

#splitlines：按照换行符进行分割
txt = "I\nlove\nyou\nso\nmuch" #\n:就是换行，打印的时候就是换行
print(txt)
l = txt.splitlines() #按换行符进行分割，分割后删除换行符，并且返回一个列表，用于数据读取后的处理
print(l)

#isalpha:判断是否是纯字母 isdigit：判断是否是纯数字 isalnum：判断是否是字母和数字的组合
a = "123qwe"
if a.isdigit():
    print("是数字")
elif a.isalpha():
    print("是字母")
elif a.isalnum():
    print("是组合")
elif a.isspace():
    print("是纯空格")

#join：
#将下面列表的元素变为字符串
a = ["aaa", "bbbb", "ccc"]
b = "_"
print(b.join(a))

b = " "   #b是空格
print(b.join(a))

b = ""     #b是无间隔
print(b.join(a))
#总结：b是什么join就用什么将原来的列表中的元素连接成字符串

#作业:去掉下面字符串中的所有\t和空格,并将切割后的列表转成字符串
test = "I have \tfall in \tlove wi\tth \tyou "
txt = test.split(" ") #删除所有空格
print(txt)
txt = test.split(" \t") #只删除空格和\t相邻的这种情况
print(txt)
txt = test.split() #默认切除所有空白字符，换行符，空格等
print(txt)
n = ""
song = n.join(txt)
print(song)

"""






