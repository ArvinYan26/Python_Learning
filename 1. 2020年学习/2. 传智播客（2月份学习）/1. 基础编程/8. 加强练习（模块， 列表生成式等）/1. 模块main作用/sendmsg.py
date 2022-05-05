def test1():
    print("----test1----")

def test2():
    print("------test2----")

#print(__name__)  #这句代码，自己调用时打印的是__main__,别人调用此模块时，打印的是这个模块的模块名

if __name__ == "__main__":
    test1()
    test2()