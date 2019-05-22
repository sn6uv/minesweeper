import glob
import os
from config import HEIGHT, WIDTH, MINES, MODEL_PATH, BATCH_SIZE, EPOCHS, PLAY_BATCH_SIZE
from game import format_move
from player import Player
from utils import ask
import numpy as np


def load_loosing_games(p):
    p.data = []
    subdir = p.get_data_subdir()
    for fname in glob.glob(os.path.join(subdir, "*.pickle")):
        print("loading ", fname)
        with open(fname, "rb") as f:
            p.load_data(f)
    print("Found", len(p.data), "examples")
    p.data = [(grid, ps) for (grid, ps) in p.data if np.sum(grid == 9) < p.mines+5]
    print("Found", len(p.data), "losing examples")


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
