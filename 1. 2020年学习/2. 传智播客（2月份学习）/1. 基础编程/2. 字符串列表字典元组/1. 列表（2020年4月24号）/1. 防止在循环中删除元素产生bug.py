a = [1, 2, 3, 4, 55, 22, 66]

#产生bug形式,删除连着的元素，第二个元素不会被删除
for i in a:
    if i == 1 or i == 2:
        a.remove(i)
print(a)

#解决bug,创建空列表，存储需要删除的元素
b = []
for i in a:
    if i == 1 or i == 2:
        b.append(i)
for i in b:
    a.remove(i)
print(a)
