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


def play_train(p, play_games, train_epochs, batch_size, save_data=False):
    '''Play a number of games and then train on the resulting data'''
    print("Playing %i games" % play_games)
    p.play(play_games)
    print("Training on %i examples" % len(p.data))
    p.train(batch_size=batch_size, epochs=train_epochs)
    if save_data:
        p.dump_data()
    p.data = []
