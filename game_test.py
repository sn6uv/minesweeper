import random

from game import Game


def test_simple_game():
  random.seed(a=0)
  g = Game(5, 5, 2)
  assert(g.mines == {(0, 2), (3, 3)})
  assert(g.guessed == set())
  assert(not g.guess((0, 4)))
  assert(g.guessed == {(1, 3), (1, 4), (2, 3), (0, 4), (0, 3), (2, 4)})
  assert(g.guess((0, 2)))

def test_full_mines():
  random.seed(a=0)
  g = Game(2, 2, 4)
  assert(g.mines == {(0, 0), (0, 1), (1, 0), (1, 1)})

def test_game_repr():
  random.seed(a=0)
  g = Game(4, 4, 1)
  assert(not g.guess((1, 1)))
  assert(repr(g) == '0000\n0000\n0011\n001 ')


if __name__ == '__main__':
  test_simple_game()
  test_full_mines()
  test_game_repr()
