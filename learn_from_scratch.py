'''Script for bootstrapping the model by playing games and trainging on them '''
from config import HEIGHT, WIDTH, MINES
from player import Player
from utils import play_train


def train_0_to_2(p):
    '''Trains to 2% win rate'''
    p.play_debug()
    play_train(p, 5000, 3, 64)
    play_train(p, 5000, 1, 64)
    p.model.save("scratch0")


def train_2_to_5(p):
    '''Trains from 2% to 5% win rate'''
    p.play_debug()
    play_train(p, 10000, 1, 64)
    p.model.save("scratch1")


def train_5_to_15(p):
    '''Trains from 5% to 10% win rate'''
    p.play_debug()
    play_train(p, 20000, 2, 256)
    p.model.save("scratch2")


def train_15_to_25(p):
    '''Trains from 5% to 10% win rate'''
    p.play_debug()
    play_train(p, 20000, 1, 256)
    play_train(p, 25000, 1, 1024)
    p.model.save("scratch3")


def train_25_to_35(p):
    '''Trains from 25% to 35% win rate'''
    p.play_debug()
    play_train(p, 25000, 1, 1024)
    play_train(p, 25000, 1, 4096)
    p.model.save("scratch4")


def train_35_to_45(p):
    '''Trains from 35% to 45% win rate'''
    p.play_debug()
    play_train(p, 50000, 3, 4096)
    p.model.save("scratch5")


def train_45_to_50(p):
    '''Trains from 45% to 50% win rate'''
    p.play_debug()
    play_train(p, 50000, 3, 16384)
    p.model.save("scratch6")


def main():
    p = Player(HEIGHT, WIDTH, MINES)

    train_0_to_2(p)
    # p.model.restore("scratch0")

    train_2_to_5(p)
    # p.model.restore("scratch1")

    train_5_to_15(p)
    # p.model.restore("scratch2")

    train_15_to_25(p)
    # p.model.restore("scratch3")

    train_25_to_35(p)
    # p.model.restore("scratch4")

    train_35_to_45(p)
    # p.model.restore("scratch5")

    train_45_to_50(p)
    # p.model.restore("scratch6")

    while True:
        p.play_debug()
        play_train(p, 50000, 1, 32768)
        p.model.save("scratch7")


if __name__ == '__main__':
    main()
