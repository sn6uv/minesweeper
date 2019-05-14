import pickle
import random

import numpy as np
import tensorflow as tf
from game import Game, format_move
from model import Model


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
    first = True
    while not hit:
      game_input = self.get_model_input(game.view())
      pred = self.predict_mines(game_input)
      pos = np.unravel_index(np.argmin(pred), (self.height, self.width))
      if first:
        # Randomise first guess to prevent bias, since first mine moves.
        pos = random.randint(0, self.height-1), random.randint(0, self.width-1)
        first = False
      if debug:
        print(format_move(game, pos, risk_matrix=pred.reshape(self.height, self.width)))
        print("p in [%f, %f]" % (np.min(pred), np.max(pred[pred<1])))
      hit = game.guess(pos)
      assert(hit is not None)
      self.data.append((game.view(), game.mines))
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
    i = [(self.get_model_input(row[0]), self.get_model_output(row[1])) for row in self.data]
    self.model.train(i, *args, **kwargs)

  def dump_data(self, f):
    pickle.dump(self.data, f)
    self.data = []

  def load_data(self, f):
    self.data.extend(pickle.load(f))

  def get_model_input(self, view):
    o = np.zeros((self.height, self.width, 10))
    for i in range(self.height):
      for j in range(self.width):
        pos = (i, j)
        if view[i][j] is None:
          o[i, j, 9] = 1
        else:
          o[i, j, view[i][j]] = 1
    return o.flatten()

  def get_model_output(self, mines):
    o = np.zeros((self.height, self.width))
    for m in mines:
      o[m] = 1
    return o.flatten()
