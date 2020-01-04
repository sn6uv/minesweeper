import os
import pickle
import random
import datetime

import numpy as np
import tensorflow as tf
from config import DATA_DIR
from game import Game, format_move
from model import Model
from symmetry import dihedral


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
            print(datetime.datetime.now(), "Win rate: %f%%" % (100.0 * won / float(rounds)))

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
        game_input = self.get_model_input(view)
        pred = self.model.predict(game_input)[0]
        pred[view.flatten()!=9]=1    # ignore alreday guessed locations
        return pred

    def train(self, *args, **kwargs):
        i = []
        for row in self.data:
            j = random.randint(0, 8)  # symmetry index
            i.append((self.get_model_input(row[0], j), self.get_model_output(row[1], j)))
        self.model.train(i, *args, **kwargs)

    def dump_data(self, f):
        pickle.dump(self.data, f)
        self.data = []

    def load_data(self, f):
        self.data.extend(pickle.load(f))

    def get_model_input(self, view, symmetry_index=0):
        x = dihedral(view, symmetry_index)
        return np.eye(10, dtype=np.int8)[x].flatten()

    def get_model_output(self, mines, symmetry_index=0):
        o = np.zeros((self.height, self.width))
        for m in mines:
            o[m] = 1
        o = dihedral(o, symmetry_index)
        return o.flatten()

    def get_data_subdir(self):
        return os.path.join(DATA_DIR, str(self.height) + '_' + str(self.width) + '_' + str(self.mines))
