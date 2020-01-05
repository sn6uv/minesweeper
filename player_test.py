import numpy as np
import random

from player import Player
from game import Game


def test_player():
    random.seed(a=0)
    p = Player(3, 3, 1)
    p.play_debug()
    assert(p.data)
    p.play(1000)
    p.train(epochs=5, batch_size=16)
    p.play(1000)
    p.play_debug()


if __name__ == '__main__':
    test_player()
