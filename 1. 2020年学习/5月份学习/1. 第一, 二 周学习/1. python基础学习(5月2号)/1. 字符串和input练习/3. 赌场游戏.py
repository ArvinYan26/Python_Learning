"""
掷筛子
1. 欢迎进入、、、、游戏
2. 输入姓名，默认用户没有必币
3. 提示用户充值买币，（100块钱30币，充值100倍数，充值不成功可以再次充币）
4. 玩一局游戏两个币，猜大小（系统随机模拟筛子产生两个数）
5. 只要猜对就奖励一个币，可以继续玩（可以自动退出，也可以没币乐了自动退出）
"""
import random

print("*"*30)
print("欢迎进入澳门赌场")
print("*"*30)

username = input("请输入用户名：")
money = 0
answer = input("确定进入游戏吗（y/n）?")
if answer == "y":
    while money < 2:
        n = int(input("金币不足，请充值（100块钱30币，充值100倍数，充值不成功可以再次充币）："))
        if n%100 ==0 and n > 0:
            money = (n//100)*30
    print("当前剩余游戏币是：{}， 玩一次扣除2个币".format(money))
    print("正在进入游戏、、、、")
    while True:
        #制筛子，随机数产生
        t1 = random.randint(1, 6)
        t2 = random.randint(1, 6)

        money -= 2
        #两个筛子的值大于6是大，否则小
        print("系统洗牌完毕，请猜大小")
        guess = input("请输入大小或者小：")
        #判断
        if ((t1+t2)>6 and guess=='大') or ((t1+t2<=6) and guess=='小'):
            print("恭喜{}！本局游戏获得奖励1个游戏币！".format(username))
            money += 1
        else:
            print("很遗憾！本局游戏输啦")

        answer = input("是否再来一局游戏，扣除2个游戏币？（y/n）")
        if answer != "y" or money <2:
            print("游戏退出、、")
            break #终止循环