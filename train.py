import glob
import os
from config import HEIGHT, WIDTH, MINES, MODEL_PATH, BATCH_SIZE, EPOCHS
from player import Player
from utils import ask


def load_data_and_train(p):
    p.data = []
    subdir = p.get_data_subdir()
    for fname in glob.glob(os.path.join(subdir, "*.pickle")):
        print("loading ", fname)
        with open(fname, "rb") as f:
            p.load_data(f)
    p.train(batch_size=BATCH_SIZE, epochs=EPOCHS)
    p.data = []


if __name__ == '__main__':
    p = Player(HEIGHT, WIDTH, MINES)

    if ask("Load saved model"):
        p.model.restore(MODEL_PATH)

    print("Training...")
    try:
        load_data_and_train(p)
    except KeyboardInterrupt:
        print("Training interrupted!")

    if ask("Would you like to save model"):
        p.model.save(MODEL_PATH)
