"""
捕鱼达人
请输入参与者名字
输入密码：
充值：500
"""
print("*"*30)
print("捕鱼达人")
print("*"*30)
username = input("请输入账户名称：")
password = input("请输入密码：")
#if username == "闫江龙" and password == 123:
print("%s请充值！" % username)
coins = int(input("请充值： ")) #input键盘输入的都是字符串类型，所以想要数字，需要转换类型
print("%s充值成功，余额为：%d"%(username, coins))
