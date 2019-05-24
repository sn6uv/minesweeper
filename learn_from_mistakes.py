import glob
import os
import random
from config import HEIGHT, WIDTH, MINES, MODEL_PATH, BATCH_SIZE, EPOCHS, PLAY_BATCH_SIZE
from game import format_move
from player import Player
from utils import ask, get_data_files
import numpy as np


def randomly_include_grid(unguessed, mines, size):
    '''Randomly include an example based on how many unguessed squares remain.

    Note that unguessed-mines is the number of squares which need to be
    guessed still. So, the ratio

    (unguessed - mines) / size

    Represents remaining game progress; it tends to 0 when unguessed == mines)
    and is mines/size (a small, non-zero value) at the start of the game.'''
    return size * random.random() > (unguessed - mines)


def load_loosing_games(p):
    size = p.height * p.width
    p.data = []
    for fname in get_data_files():
        print("loading ", fname)
        with open(fname, "rb") as f:
            p.load_data(f)
    print("Found", len(p.data), "examples")
    p.data = [(grid, ps) for (grid, ps) in p.data if randomly_include_grid(np.sum(grid == 9), p.mines, size)]
    print("Chose", len(p.data), "examples")


if __name__ == '__main__':
    p = Player(HEIGHT, WIDTH, MINES)

    if ask("Load saved model"):
        p.model.restore(MODEL_PATH)

    load_loosing_games(p)

    try:
        p.train(batch_size=BATCH_SIZE, epochs=EPOCHS)
    except KeyboardInterrupt:
        print("Training interrupted!")

    p.play(PLAY_BATCH_SIZE)
    for _ in range(5):
        p.play(1, debug=True)
        print()

    if ask("Would you like to save model"):
        p.model.save(MODEL_PATH)
