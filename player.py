import os
import pickle
import random
import datetime

import numpy as np
import tensorflow as tf
from config import DATA_DIR, GAME_BATCH_SIZE
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

    def play_debug(self):
        g = Game(self.height, self.width, self.mines)
        self.play_game(g, True)

    def play(self, rounds, game_batch_size=GAME_BATCH_SIZE):
        won = 0
        gs = [Game(self.height, self.width, self.mines) for _ in range(game_batch_size)]
        started_games = game_batch_size
        while gs:
            views = np.array([g.view() for g in gs])
            preds = self.predict_mines(views)
            for i, g in enumerate(gs):
                hit_mine = self.play_move(g, preds[i, :], False)
                self.data.append((g.view(), g.mines))
                if g.is_won():
                    won += 1
                elif not hit_mine:
                    continue

                if started_games < rounds:
                    gs[i] = Game(self.height, self.width, self.mines)
                    started_games += 1
                else:
                    gs[i] = None
            gs = [g for g in gs if g is not None]
        print(datetime.datetime.now(), "Win rate: %f%%" % (100.0 * won / float(rounds)))

    def play_game(self, game, debug=False):
        hit_mine = False
        while not hit_mine:
            pred = self.predict_mines(game.view())
            hit_mine = self.play_move(game, pred, debug)
            self.data.append((game.view(), game.mines))
            if game.is_won():
                if debug:
                    print("Won!")
                return True
        if debug:
            print("Lost!")
        return False

    def play_move(self, game, pred, debug):
        view = game.view()
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
        pred = self.model.predict(game_input)

        # Ignore alreday guessed locations
        pred[view.reshape(*view.shape[:-2], -1) != 9] = 1

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
        onehot = np.eye(10, dtype=np.int8)[x]
        # Flatten final 3 axes (category, x, y)
        return onehot.reshape(*onehot.shape[:-3], -1)

    def get_model_output(self, mines, symmetry_index=0):
        o = np.zeros((self.height, self.width))
        for m in mines:
            o[m] = 1
        o = dihedral(o, symmetry_index)
        return o.flatten()

    def get_data_subdir(self):
        return os.path.join(DATA_DIR, str(self.height) + '_' + str(self.width) + '_' + str(self.mines))
