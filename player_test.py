import numpy as np
import random

from player import Player
from game import Game


def test_player():
    random.seed(a=0)
    p = Player(3, 3, 1)
    p.play(10, debug=True)
    assert(p.data)
    p.train()


if __name__ == '__main__':
    test_player()
