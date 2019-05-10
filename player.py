import numpy as np
from game import Game

def get_input(game):
  o = np.zeros((game.height, game.width))
  for i in range(game.height):
    for j in range(game.width):
      pos = (i, j)
      if pos not in game.guessed:
        o[pos] = 10
        continue
      o[pos] = game.count_nearby_mines(pos)
  return o

def get_output(game):
  o = np.zeros((game.height, game.width))
  for m in game.mines:
    o[m] = 1
  return o

class Player:
  '''Plays minesweeper'''

  def __init__(self, height, width, mines):
    self.height = height
    self.width = width
    self.mines = mines
    self.data = []

  def play(self, rounds):
    for _ in range(rounds):
      g = Game(self.height, self.width, self.mines)
      self.play_game(g)

  def play_game(self, game):
    hit = False
    while not hit:
      game_input = get_input(game)
      pred = self.predict_mines(game_input)
      pos = np.unravel_index(np.argmin(pred), pred.shape)
      hit = game.guess(pos)
      if hit is None:
        print("Invalid choice")
        break
      self.data.append((game_input, pred, get_output(game)))

  def predict_mines(self, game_input):
    pred = np.random.random((self.height, self.width))
    # Set already guessed positions to 1 to avoid choosing them
    pred[game_input < 10] = 1
    return pred
