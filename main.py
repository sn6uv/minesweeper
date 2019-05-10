from player import Player

p = Player(3, 3, 1)
for _ in range(10):
  # drop old data
  if (len(p.data) > 10000):
    p.data = p.data[len(p.data)-10000:]

  # play some more games and train
  p.play(1000)
  p.train()
