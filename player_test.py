import numpy as np
import random

from player import Player, get_model_input, get_model_output
from game import Game


def test_player():
  random.seed(a=0)
  p = Player(3, 3, 1)
  p.play(1)
  assert(p.data)


def test_get_model_input():
  random.seed(a=0)
  g = Game(2, 2, 1)
  g.guess((1,1))
  output = np.zeros(40)
  output[9] = 1.0
  output[19] = 1.0
  output[29] = 1.0
  output[31] = 1.0
  assert(np.array_equal(get_model_input(g), output))


def test_get_model_output():
  random.seed(a=0)
  g = Game(2, 2, 1)
  assert(np.all(get_model_output(g) == np.array([0, 0, 0, 1])))


if __name__ == '__main__':
  test_player()
  test_get_model_input()
  test_get_model_output()
