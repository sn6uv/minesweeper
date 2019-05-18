import os
import datetime
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
    p = Player(9, 9, 10)
    model_path = "models/9_9_1/model.ckpt"

    if ask("Load saved model"):
        p.model.restore(model_path)

    print("Playing...")
    try:
        for i in range(100):
            print("Round %3i" % i)
            p.play(5000)
            p.play(1, debug=True)
            dump_data(p)
    except KeyboardInterrupt:
        print("Playing interrupted!")
