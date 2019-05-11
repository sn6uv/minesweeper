from player import Player
from player import Player

p = Player(6, 15, 10)
for _ in range(1000):
  p.play(1000)
  p.train()
  p.play(1, debug=True)
