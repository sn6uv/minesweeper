import glob
import os
import random
from config import HEIGHT, WIDTH, MINES, MODEL_DIR


def ask(question):
    while True:
        i = input(question + " [y/n] ")
        if i and i in 'yY':
            return True
        if i and i in 'nN':
            return False


def get_data_dir():
    return os.path.join('data', str(HEIGHT) + '_' + str(WIDTH) + '_' + str(MINES))


def get_data_files():
    subdir = get_data_dir()
    fnames = glob.glob(os.path.join(subdir, "*.pickle"))
    random.shuffle(fnames)
    return fnames


def model_path(name):
    return os.path.join(MODEL_DIR, "%i_%i_%i/%s.ckpt" % (HEIGHT, WIDTH, MINES, name))
