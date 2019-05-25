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


def train_0_to_1(p):
    '''Trains to 1.5% win rate'''
    p.play(1, debug=True)
    play_train(p, 1000, 3, 64)
    play_train(p, 5000, 3, 64)
    p.data = []
    p.model.save("models/9_9_10_scratch0/model.ckpt")


def train_1_to_5(p):
    '''Trains from 1.5% to 5% win rate'''
    p.play(1, debug=True)
    play_train(p, 10000, 2, 64)
    p.data = []
    p.model.save("models/9_9_10_scratch1/model.ckpt")


def train_5_to_15(p):
    '''Trains from 5% to 10% win rate'''
    p.play(1, debug=True)
    play_train(p, 10000, 2, 64)
    play_train(p, 25000, 2, 256)
    p.data = []
    p.model.save("models/9_9_10_scratch2/model.ckpt")


def train_15_to_25(p):
    '''Trains from 5% to 10% win rate'''
    p.play(1, debug=True)
    play_train(p, 20000, 1, 256)
    play_train(p, 25000, 2, 1024)
    p.data = []
    p.model.save("models/9_9_10_scratch3/model.ckpt")


def main():
    p = Player(HEIGHT, WIDTH, MINES)

    train_0_to_1(p)
    # p.model.restore("models/9_9_10_scratch0/model.ckpt")

    train_1_to_5(p)
    # p.model.restore("models/9_9_10_scratch1/model.ckpt")

    train_5_to_15(p)
    # p.model.restore("models/9_9_10_scratch2/model.ckpt")

    train_15_to_25(p)
    # p.model.restore("models/9_9_10_scratch3/model.ckpt")

    p.play(1, debug=True)
    p.play(1000)


if __name__ == '__main__':
    main()
