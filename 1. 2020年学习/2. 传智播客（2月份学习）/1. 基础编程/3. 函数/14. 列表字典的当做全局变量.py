#列表字典当做全局变量的时候不用加global声明可以直接在函数中用
#如果加了，就会让别人清楚地知道，你用的一定是全局变量
nums = [11, 22, 33]
infor = {"name":"老王"}

def test():
    nums.append(44)
    infor["age"] = 18

def test2():
    print(nums)
    print(infor)

test()
test2()