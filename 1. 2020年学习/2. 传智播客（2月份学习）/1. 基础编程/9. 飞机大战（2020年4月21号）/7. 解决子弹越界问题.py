import pygame
from pygame.locals import *
import time

class HeroPlane(object):
    def __init__(self, screen_temp): #__init__方法，创建对象的时候就会直接调用
        self.x = 210
        self.y = 700
        self.screen = screen_temp
        self.image = pygame.image.load("./feiji/hero1.png")
        self.bullet_list = [] #存储发射出来的子弹对象引用

    def display(self):
        self.screen.blit(self.image, (self.x, self.y))
        #显示飞机的同时让所有子弹也显示， 用for循环
        for bullet in self.bullet_list:
            bullet.display()
            bullet.move() #每次发射子弹都自动向上移动，所以子弹需要有move方法
            #在for循环中删除连续的元素时有bug，这里是因为现实的时候是死循环，一遍一遍的删除，所以才没出现
            if bullet.judge(): #判断子弹是否越界,子弹中的方法，所以去子弹类中找此方法
                self.bullet_list.remove(bullet)

    def move_left(self):
        self.x -= 5

    def move_right(self):
        self.x += 5
    def fire(self):
        """开火，需要先创建子弹对象"""
        self.bullet_list.append(Bullet(self.screen, self.x, self.y)) #创建子弹对象

class EnemyPlane(object):
    """敌机飞机"""
    def __init__(self, screen_temp):
        self.x = 0  #敌机坐标
        self.y = 0
        self.screen = screen_temp  #显示敌机
        self.image = pygame.image.load("./feiji/enemy0.png") #背景图片
        #self.bullet_list = [] #存储发射出来的子弹对象引用
        self.direction = "right" #用来存储飞机默认的显示方向

    def display(self):
        self.screen.blit(self.image, (self.x, self.y))
        #显示飞机的同时让所有子弹也显示， 用for循环
        """
        for bullet in self.bullet_list:
            bullet.display()
            bullet.move() #每次发射子弹都自动向上移动，所以子弹需要有move方法
        """
    def move(self):
        #判断敌机移动方向
        if self.direction == "right":
            self.x += 5
        elif self.direction == "left":
            self.x -= 5
        #判断什么时候左右截止
        if self.x > 480-50:
            self.direction = "left"
        elif self.x < 0:
            self.direction = "right"

    def fire(self):
        """开火，需要先创建子弹对象"""
        self.bullet_list.append(Bullet(self.screen, self.x, self.y)) #创建子弹对象

class Bullet(object):
    def __init__(self, screen_temp, x, y):
        self.x = x+40
        self.y = y-20
        self.screen = screen_temp
        self.image = pygame.image.load("./feiji/bullet.png")

    def display(self):
        self.screen.blit(self.image, (self.x, self.y))

    def move(self):
        self.y -= 5

    def judge(self):
        if self.y < 0:  #如果想验证，改成200，就可以，多动脑子
            return True #返回真就执行，假就不执行
        else:
            return False

def key_control(hero_temp):  # 找一个形参hero_temp接受实参hero
    # 获取事件，比如按键等 ，pygame里面的功能，不用管
    for event in pygame.event.get():
        # 判断是否点击了退出按钮
        if event.type == QUIT:
            print("exit")
            exit()
        # 判断是否是按下了键
        elif event.type == KEYDOWN:
            # 检测按键是否是a或者left
            if event.key == K_a or event.key == K_LEFT:
                print('left')
                hero_temp.move_left()

            # 检测按键是否是d或者right
            elif event.key == K_d or event.key == K_RIGHT:
                print('right')
                hero_temp.move_right()

            # 检测按键是否是空格键
            elif event.key == K_SPACE:
                print('space')
                hero_temp.fire()

def main():
    # 1. 创建一个窗口，用来显示内容
    screen = pygame.display.set_mode((480, 800), 0, 32)
    # 2. 创建一个和窗口大小的图片，用来充当背景
    background = pygame.image.load("./feiji/background.png").convert()
    # 3. 创建一个飞机对象
    hero = HeroPlane(screen) #调用screen，传给screen_temp
    #创建一个敌机
    enemy = EnemyPlane(screen)
    while True:  # 表示死循环
        screen.blit(background, (0, 0))
        hero.display()
        enemy.display()
        enemy.move()
        pygame.display.update()
        key_control(hero)  # hero是实参

        time.sleep(0.01) # 延时，让程序运算速度降下来，也会影响子弹速度

if __name__ == "__main__":

    main()





