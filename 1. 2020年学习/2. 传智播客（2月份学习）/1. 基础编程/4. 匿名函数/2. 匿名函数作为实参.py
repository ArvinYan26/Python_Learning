def test(a,b,func):
    result = func(a,b)
    return result

ret = test(3, 4, lambda x,y:x+y)
print(ret)