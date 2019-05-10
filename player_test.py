import random

from player import Player

def test_player():
  random.seed(a=0)
  p = Player(3, 3, 1)
  p.play(1)
  assert(p.data)
  

if __name__ == '__main__':
  test_player()
