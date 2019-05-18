import glob
import os
from player import Player
from utils import ask


def load_data_and_train(p):
    p.data = []
    subdir = p.get_data_subdir()
    for fname in glob.glob(os.path.join(subdir, "*.pickle")):
        print("loading ", fname)
        with open(fname, "rb") as f:
            p.load_data(f)
    p.train(batch_size=64)
    p.data = []


if __name__ == '__main__':
    p = Player(9, 9, 10)
    model_path = "models/9_9_10/model.ckpt"

    if ask("Load saved model"):
        p.model.restore(model_path)

    print("Training...")
    try:
        load_data_and_train(p)
    except KeyboardInterrupt:
        print("Training interrupted!")

    if ask("Would you like to save model"):
        p.model.save(model_path)
