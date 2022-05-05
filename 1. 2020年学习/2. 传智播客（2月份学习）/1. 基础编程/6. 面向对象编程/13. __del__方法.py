class Dog:
    def __del__(self): #程序结束或者所有引用被删除，释放内存时都会触发这个方法
        print("-----英雄over-----")

dog1 = Dog()
dog2 = dog1

del dog1   #相当于只删除了dog1这个引用，不影响dog2这个引用的使用
del dog2
#当两个引用都被删除，原来的内存就会被释放，python解释器就会自动调用类中的__del__方法
#然后打印英雄over，
print("=================")