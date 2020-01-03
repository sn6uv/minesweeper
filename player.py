import os
import pickle
import random

import numpy as np
import tensorflow as tf
from game import Game, format_move
from model import Model


class Player:
    '''Plays minesweeper'''

    def __init__(self, height, width, mines):
        self.height = height
        self.width = width
        self.mines = mines
        self.data = []
        self.model = Model(height, width)

    def play(self, rounds, debug=False):
        won = 0
        for _ in range(rounds):
            g = Game(self.height, self.width, self.mines)
            won += self.play_game(g, debug)
        if rounds > 1:
            print("Win rate: %f%%" % (100.0 * won / float(rounds)))

    def play_game(self, game, debug=False):
        hit_mine = False
        while not hit_mine:
            hit_mine = self.play_move(game, debug)
            self.data.append((game.view(), game.mines))
            if game.is_won():
                if debug:
                    print("Won!")
                return True
        if debug:
            print("Lost!")
        return False

    def play_move(self, game, debug):
        view = game.view()
        pred = self.predict_mines(view)
        pos = np.unravel_index(np.argmin(pred), (self.height, self.width))
        if not game.guessed:
            # Randomise first guess to prevent bias, since first mine moves.
            pos = random.randint(
                0, self.height-1), random.randint(0, self.width-1)
        if debug:
            print(format_move(game.view(), game.mines, pos, risk_matrix=pred.reshape(
                self.height, self.width)))
            print("p in [%f, %f]" % (np.min(pred), np.max(pred[pred < 1])))
        hit = game.guess(pos)
        assert(hit is not None)
        return hit

    def predict_mines(self, view):
        pred = self.model.predict(self.get_model_input(view))[0]
        pred[view.flatten()!=9]=1    # ignore alreday guessed locations
        return pred

    def train(self, *args, **kwargs):
        i = [(self.get_model_input(row[0]), self.get_model_output(row[1]))
             for row in self.data]
        self.model.train(i, *args, **kwargs)

    def dump_data(self, f):
        pickle.dump(self.data, f)
        self.data = []

    def load_data(self, f):
        self.data.extend(pickle.load(f))

    def get_model_input(self, view):
        return tf.keras.utils.to_categorical(view)

    def get_model_output(self, mines):
        o = np.zeros((self.height, self.width))
        for m in mines:
            o[m] = 1
        return o.flatten()

    def get_data_subdir(self):
        return os.path.join('data', str(self.height) + '_' + str(self.width) + '_' + str(self.mines))
