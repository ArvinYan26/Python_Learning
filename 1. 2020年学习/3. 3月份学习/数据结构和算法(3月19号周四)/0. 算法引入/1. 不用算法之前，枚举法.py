#如果 a+b+c=1000，且 a^2+b^2=c^2（a,b,c 为自然数），如何求出所有a、b、c可能的组合?
import time
start_time = time.time()
for a in range(1001):
    for b in range(1001):
        for c in range(1001):
            if a+b+c==1000 and a**2+b**2==c**2:
                print("a, b, c: %d, %d, %d"%(a, b, c))
end_time = time.time()
print("用时：%f"%(end_time-start_time)) #%f:浮点实数，用115.86706秒
print("运行结束")