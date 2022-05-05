#用来存储名片
card_infors = []
def print_menu():
    """"用来存储名片"""
    print("="*50)
    print("   名片管理系统 V0.01")
    print(" 1. 添加一个新的名片")
    print(" 2. 删除一个名片")
    print(" 3. 修改一个名片")
    print(" 4. 查询一个名片")
    print(" 5. 显示所有的名片")
    print(" 6. 保存信息")
    print(" 7. 退出系统")
    print("="*50)

def add_new_card_infor():
    """完成添加新名片"""
    new_name = input("请输入新的名字:")
    new_qq = input("请输入新的QQ:")
    new_weixin = input("请输入新的微信:")
    new_addr = input("请输入新的住址:")

    # 定义一个新的字典,用来存储一个新的名片
    new_infor = {}
    new_infor['name'] = new_name
    new_infor['qq'] = new_qq
    new_infor['weixin'] = new_weixin
    new_infor['addr'] = new_addr

    # 将一个字典,添加到列表中
    global card_infors
    card_infors.append(new_infor)

def find_card_infor():
    """"用来查询一个名片"""
    global card_infors
    find_name = input("请输入要查找的名字：")
    find_flag = 0 #默认表示没有找到
    for temp in card_infors:
        if find_name == temp["name"]:
            print("%s\t%s\t%s\t%s"%(temp['name'], temp['qq'], temp['weixin'], temp['addr']))
            find_flag = 1 #表示找到了
            break
    #判断是否找到了
    if find_flag == 0:
        print("没有此人信息.....")

def show_all_infor():
    """显示所有名片信息"""
    global card_infor
    print("姓名\tQQ\t微信\t住址")
    for temp in card_infors:
        print("%s\t%s\t%s\t%s"%(temp['name'], temp['qq'], temp['weixin'], temp['addr']))

def save_2_file():
    """把已经添加的信息保存到文件中"""

    f = open("backup.data", "w") #读写，.data：自己定义的格式，防止别人更改内容
    f.write(str(card_infors))
    f.close()

def load_file():
    global card_infors
    try:
        f = open("backup.data")
        card_infors = eval(f.read()) #文件转成字符串之前什么样子，eval就恢复到什么形式
        f.close()
    except Exception:
        pass

def main():
    """完成所有函数的调用"""
    #恢复之前的数据到程序中
    #1. 打印功能提示
    print_menu()

    while True:

        #2. 获取用户的输入
        num = int(input("请输入操作序号:"))

        #3. 根据用户的数据执行相应的功能
        if num==1:
            add_new_card_infor()
        elif num==2:
            pass
        elif num==3:
            pass
        elif num==4:
            find_card_infor()
        elif num==5:
            show_all_infor()

        elif num == 6:
            save_2_file()

        elif num==7:
            break
        else:
            print("输入有误,请重新输入")


        print("")

#调用主函数（c语言里开始必须要写main（）函数，不写执行不了）
if __name__ == "__main__":
    main()