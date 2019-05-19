import os
import datetime
from config import HEIGHT, WIDTH, MINES, MODEL_PATH, PLAY_BATCH_SIZE, PLAY_ROUNDS
from player import Player
from utils import ask


def dump_data(p):
    fname = str(datetime.datetime.now()) + ".pickle"
    subdir = p.get_data_subdir()
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    with open(os.path.join(subdir, fname), 'wb') as f:
        p.dump_data(f)
    print("Saved data %s" % fname)


if __name__ == '__main__':
    p = Player(HEIGHT, WIDTH, MINES)

    if ask("Load saved model"):
        p.model.restore(MODEL_PATH)

    print("Playing...")
    try:
        for i in range(PLAY_ROUNDS):
            print("Round %3i" % i)
            p.play(PLAY_BATCH_SIZE)
            p.play(1, debug=True)
            dump_data(p)
    except KeyboardInterrupt:
        print("Playing interrupted!")
