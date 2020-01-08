'''Script for training the model by playing games and trainging on them '''
from config import HEIGHT, WIDTH, MINES
from player import Player
from utils import play_train


def main():
    p = Player(HEIGHT, WIDTH, MINES)
    p.model.restore("pretrained")
    while True:
        p.play_debug()
        play_train(p, 50000, 1, 32768)
        p.model.save("trained")


if __name__ == '__main__':
    main()
