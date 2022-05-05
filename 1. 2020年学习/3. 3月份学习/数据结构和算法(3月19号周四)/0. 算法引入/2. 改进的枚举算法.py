import time
start_time = time.time()
for a in range(1001):
    for b in range(1001):
        c = 1000-a-b
        if a**2+b**2==c**2:
            print("a, b , c: %d, %d, %d"%(a, b, c))
end_time = time.time()
print("用时: %f"%(end_time-start_time)) #用时1秒
print("运行结束")

