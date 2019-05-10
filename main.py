from player import Player

p = Player(9, 9, 10)
for _ in range(100):
  # drop old data
  if (len(p.data) > 500000):
    p.data = p.data[len(p.data)-500000:]

  # play some more games and train
  p.play(1000)
  p.train()
