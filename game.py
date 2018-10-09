import random
import copy
from collections import Counter
import numpy as np


class Node():
    def __init__(self, position, father):
        self.x, self.y = position
        self.father = father
        self.status = 1

    def explore(self, board, link, visited):
        # 确定下一个探索点,进入下一个状态
        x, y = self.x, self.y
        check = None
        size = len(board)
        if self.status == 1 and self.x != size - 1:
            check = [x + 1, y]
        if self.status == 2 and self.y != size - 1:
            check = [x, y + 1]
        if self.status == 3 and self.x != 0:
            check = [x - 1, y]
        if self.status == 4 and self.y != 0:
            check = [x, y - 1]
        self.status = self.status + 1

        # 如果下一个探索点不存在或者下一个探索点已经在链表中,返回false,指示继续循环
        # 否则,如果下一个探索点为同色,返回check,指示退出循环;否则返回false,指示继续循环
        if check is None:
            # print self.status-1,self.x,',',self.y," edge"
            return False
        exist = (check[0], check[1]) in visited
        if exist:
            # print self.status-1,self.x,',',self.y," exist"
            return False
        if board[0][0] == board[check[0]][check[1]]:
            # print self.status-1,self.x,',',self.y," child ",check
            return check
        else:
            # print self.status-1,self.x,',',self.y," not same"
            return False

    def next(self, board, link, visited):
        check = False
        while (self.status != 5):
            check = self.explore(board, link, visited)
            if (check):
                break
        return check


class Spider():
    def __init__(self, board):
        # 定义链表,初始化蜘蛛在左上角第一个点
        self.link = [Node([0, 0], None)]
        self.visited = set([(0, 0)])
        self.spider = self.link[0]

    def clean(self):
        # 所有节点为未检查
        for n in self.link:
            n.status = 1

    def nodeExist(self, child):
        # 检查节点是不是已经存在于链表中
        return (child.x, child.y) in self.visited

    def target_board(self, board):
        # 定义链表,初始化蜘蛛在左上角第一个点
        size = len(board)
        self.link = [Node([0, 0], None)]
        self.visited = set([(0, 0)])
        self.spider = self.link[0]
        # target_board = [[0 for x in range(size)] for y in range(size)]
        target_board = np.zeros(shape=(size, size), dtype=np.int)
        self.clean()
        next = self.spider.next(board, self.link, self.visited)
        # 只要不是spider在原点且检查完了所有方向(这意味着整个棋盘已经检查完毕),就继续检查
        while (not (self.spider.father == None and next == False)):
            if (next):  # 若是存在下一个节点,则新建这个节点插入到链表里面去(如果其不存在于链表中的话),蜘蛛来到新节点
                father = self.spider
                child = Node(next, father)
                if (not self.nodeExist(child)):
                    self.link.append(child)
                    self.visited.add((child.x, child.y))
                self.spider = child
            else:  # 否则就让蜘蛛回到父节点去
                self.spider = self.spider.father
            next = self.spider.next(board, self.link, self.visited)
        # 最后,将每一个节点所对应的(x,y坐标设为target point)
        for n in self.link:
            target_board[n.x, n.y] = 1
        return target_board


class Game():
    def __init__(self, need_cal_f=False, size=12):
        # init mainBoard
        self.size = size
        point_num = self.point_num = size * size
        if point_num % 6 != 0:
            raise ("size * size can not devide 6!")

        self.main_borad = np.zeros(shape=(size, size), dtype=np.int)
        posList = []
        for x in range(size):
            for y in range(size):
                posList.append([x, y])
        color = 1
        left = point_num - 1
        for x in range(6):
            for y in range(point_num // 6):
                k = random.randint(0, left)
                i, j = posList[k]
                self.main_borad[i][j] = color
                posList[k] = posList[left]
                left = left - 1
            color = color + 1
        # print(self)
        # self.main_borad = np.array(self.main_borad)

        # init start
        self.start = ''
        for x in range(size):
            for y in range(size):
                self.start += str(self.main_borad[x, y])

        # init all step
        self.all_step = ''

        # init spider
        self.spider = Spider(self.main_borad)

        # init target_board
        self.target_board = self.spider.target_board(self.main_borad)
        # print self.target_board
        # init step
        self.step = 0

        # init baseCorlor
        self.base_color = self.main_borad[0, 0]

        # calculate f value
        self.need_cal_f = need_cal_f
        self.cal_f()

    @property
    def target_area(self):
        return len(np.nonzero(self.target_board)[0])

    def cal_f(self):
        if not self.need_cal_f:
            return
        board = np.reshape(self.main_borad, (self.point_num))
        # print(colors)
        remain_color = Counter(board)

        # self.f = len(remain_color)
        # return

        smallest_manhattan_distance = self.size * 2
        for x in range(self.size):
            for y in range(self.size):
                if (self.target_board[x, y] == 1):
                    manhattan_distance = self.size - 1 - x + self.size - 1 - y
                    if manhattan_distance < smallest_manhattan_distance:
                        smallest_manhattan_distance = manhattan_distance

        self.f = self.step + max(smallest_manhattan_distance, len(remain_color) - 1) + \
            (144 - self.target_area) * 1.0 / self.point_num

    def change(self, color, start=1, visual=False):
        color = int(color) + 1 - start
        self.base_color = color
        self.step = self.step + 1
        for x in range(self.size):
            for y in range(self.size):
                if (self.target_board[x, y] == 1):
                    self.main_borad[x, y] = color
        if visual:
            print(self)
        self.all_step += str(color)
        self.target_board = self.spider.target_board(self.main_borad)
        # print self.target_board
        self.cal_f()

    def is_over(self):
        if self.target_area == self.point_num:
            return True
        else:
            return False

    def __str__(self):
        output = ''
        for x in range(self.size):
            for y in range(self.size):
                if (y != self.size - 1):
                    output += str(self.main_borad[x, y]) + "  "
                else:
                    output += str(self.main_borad[x, y])
            output += "\n"
        return output

    @property
    def board_string(self):
        output = ''
        for x in range(self.size):
            for y in range(self.size):
                output += str(self.main_borad[x, y])
        return output


if __name__ == "__main__":
    import time

    test_list = []
    start = time.time()
    for _ in range(5000):
        if _ % 1000 == 0:
            print(_)
        game = Game(size=6)

        while not game.is_over():
            color = random.randint(1, 6)
            #  color = int(raw_input())
            game.change(color, visual=False)
        test_list.append(game.step)

    print(sum(test_list) / len(test_list))
    end = time.time()

    print((end - start) / 1000)