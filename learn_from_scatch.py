'''Script for bootstrapping the model by playing games and trainging on them '''
from config import HEIGHT, WIDTH, MINES
from player import Player

MODEL_PATH = "models/9_9_10_scratch/model.ckpt"


def play_train(p, play_games, train_epochs, batch_size):
    '''Play a number of games and then train on the resulting data'''
    print("Playing %i games" % play_games)
    p.play(play_games)
    print("Training on %i examples" % len(p.data))
    p.train(batch_size=batch_size, epochs=train_epochs)
    p.data = []


def train_0_to_2(p):
    '''Trains to 2% win rate'''
    p.play(1, debug=True)
    play_train(p, 5000, 3, 64)
    play_train(p, 5000, 1, 64)
    p.model.save("models/9_9_10_scratch0/model.ckpt")


def train_2_to_5(p):
    '''Trains from 2% to 5% win rate'''
    p.play(1, debug=True)
    play_train(p, 10000, 1, 64)
    p.model.save("models/9_9_10_scratch1/model.ckpt")


def train_5_to_15(p):
    '''Trains from 5% to 10% win rate'''
    p.play(1, debug=True)
    play_train(p, 20000, 2, 256)
    p.model.save("models/9_9_10_scratch2/model.ckpt")


def train_15_to_25(p):
    '''Trains from 5% to 10% win rate'''
    p.play(1, debug=True)
    play_train(p, 20000, 1, 256)
    play_train(p, 25000, 1, 1024)
    p.model.save("models/9_9_10_scratch3/model.ckpt")


def train_25_to_35(p):
    '''Trains from 25% to 35% win rate'''
    p.play(1, debug=True)
    play_train(p, 25000, 1, 1024)
    play_train(p, 25000, 1, 4096)
    p.model.save("models/9_9_10_scratch4/model.ckpt")


def train_35_to_45(p):
    '''Trains from 35% to 45% win rate'''
    p.play(1, debug=True)
    play_train(p, 50000, 3, 4096)
    p.model.save("models/9_9_10_scratch5/model.ckpt")


def train_45_to_55(p):
    '''Trains from 45% to 55% win rate'''
    p.play(1, debug=True)
    play_train(p, 50000, 3, 16384)
    p.model.save("models/9_9_10_scratch6/model.ckpt")


def main():
    p = Player(HEIGHT, WIDTH, MINES)

    train_0_to_2(p)
    # p.model.restore("models/9_9_10_scratch0/model.ckpt")

    train_2_to_5(p)
    # p.model.restore("models/9_9_10_scratch1/model.ckpt")

    train_5_to_15(p)
    # p.model.restore("models/9_9_10_scratch2/model.ckpt")

    train_15_to_25(p)
    # p.model.restore("models/9_9_10_scratch3/model.ckpt")

    train_25_to_35(p)
    # p.model.restore("models/9_9_10_scratch4/model.ckpt")

    train_35_to_45(p)
    # p.model.restore("models/9_9_10_scratch5/model.ckpt")

    train_45_to_55(p)
    # p.model.restore("models/9_9_10_scratch6/model.ckpt")

    p.play(1, debug=True)
    p.play(1000)


if __name__ == '__main__':
    main()
