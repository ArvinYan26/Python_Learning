#python和C里面的赋值是不一样的。
#引用：当给变量赋值时，不是把值赋值给了你，而是把变量指向的地址给了你。这和C是不同的
a = 100
b = a
print(id(a))
print(id(b))

#但是C里面不是这样的，a， 和 b的地址不一样

A = [11, 22, 33]
B = A   #等号就是引用
print(id(A))
print(id(B))

A = A.append(44)
print(B)
print(A)