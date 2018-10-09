from game import Game
from copy import deepcopy
from numpy import argmax

def run(game, action, start=0):
    tmp_game = deepcopy(game)
    tmp_game.change(action, start=0)
    return tmp_game.target_area

def greedy(game, depth=None):
    if depth is not None:
        raise NotImplementedError("depth can only be one currently, sorry")

    size = game.size
    actions = range(size)
    socre = [run(game, a) for a in actions]

    return argmax(socre)

if __name__ == "__main__":
    test_list = []
    for i in range(3000):
        if i % 300 == 0:
            print(i)
        game = Game(size=6)
        while not game.is_over():
            action = greedy(game)
            game.change(action, start=0, visual=False)
    test_list.append(game.step)
    print(sum(test_list) / len(test_list))

    # avg: 12 steps