class Dog:

    #私有方法(一般是比较重要的方法，不想在外边直接被调用，想要调用，可以先通过调用其他方法，在其他方法里面去调用私有方法)
    def __send_msg(self):

        print("---正在发送短信----")

    #公有方法
    def send_msg(self, new_money):
        if new_money > 10000:
            self.__send_msg() #类里面的方法去调用另一个方法时，加self.
        else:
            print("余额不足，请及时充值---")
dog = Dog()
dog.send_msg(100)

class PhoneBill:

    def __send_msg(self):
        print("-----正在发送短信----")

    def send_msg(self, new_money):
        if new_money > 100:
            self.__send_msg()
        else:
            print("余额不足")

bill = PhoneBill()
bill.send_msg(10000)
bill.send_msg(10)