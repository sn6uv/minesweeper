import numpy as np
import tensorflow as tf
from game import Game, format_move
from model import Model

def get_input(game):
  o = np.zeros((game.height, game.width, 10))
  for i in range(game.height):
    for j in range(game.width):
      pos = (i, j)
      if pos not in game.guessed:
        o[i, j, 9] = 1
        continue
      o[i, j, game.count_nearby_mines(pos)] = 1
  return o.flatten()

def get_output(game):
  o = np.zeros((game.height, game.width))
  for m in game.mines:
    o[m] = 1
  return o.flatten()


class Player:
  '''Plays minesweeper'''

  def __init__(self, height, width, mines, model=None):
    self.height = height
    self.width = width
    self.mines = mines
    self.data = []
    if model is None:
      self.model = Model(height, width)
    else:
      self.model = model

  def play(self, rounds, debug=False):
    won = 0
    for _ in range(rounds):
      g = Game(self.height, self.width, self.mines)
      won += self.play_game(g, debug)
    if rounds > 1:
      print("Win rate: %f%%" % (100.0 * won / float(rounds)))

  def play_game(self, game, debug=False):
    hit = False
    while not hit:
      game_input = get_input(game)
      pred = self.predict_mines(game_input)
      pos = np.unravel_index(np.argmin(pred), (self.height, self.width))
      hit = game.guess(pos)
      if debug:
        print('-' * game.width)
        print(format_move(game, pos))
      assert(hit is not None)
      self.data.append((game_input, get_output(game)))
      if game.is_won():
        if debug:
          print("Won!")
        return True
    if debug:
      print("Lost!")
    return False

  def predict_mines(self, game_input):
    # pred = np.random.random(self.height * self.width)
    pred = self.model.predict(game_input)[0]
    # Set already guessed positions to 1 to avoid choosing them
    pred[game_input.reshape(self.height, self.width, 10)[:,:,9].flatten() != 1] = 1
    return pred

  def train(self, *args, **kwargs):
    self.model.train(self.data, *args, **kwargs)
