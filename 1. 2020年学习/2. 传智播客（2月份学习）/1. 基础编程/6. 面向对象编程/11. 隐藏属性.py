class Dog:
    def set_age(self, new_age):
        if new_age >= 0 and new_age <= 100:
            self.age = new_age
        else:
            self.age = 0

    def get_age(self):
        return self.age




dog = Dog()
#dog.age = 10  #如果不通过方法获取年龄，有时候是有风险的
#dog.name = "小白"

#print(dog.age)

dog.set_age(10)
age = dog.get_age()
print(age)



