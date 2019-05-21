import random
import numpy as np

from game import Game, format_move, basic_style


def test_simple_game():
    random.seed(a=0)
    g = Game(5, 5, 2)
    assert(g.mines == {(0, 2), (3, 3)})
    assert(g.guessed == set())
    assert(not g.guess((0, 4)))
    assert(g.mines == {(0, 2), (3, 3)})
    assert(g.guessed == {(1, 3), (1, 4), (2, 3), (0, 4), (0, 3), (2, 4)})
    assert(g.guess((0, 2)))
    assert(g.mines == {(0, 2), (3, 3)})


def test_no_bad_luck():
    '''Can't hit a mine on first turn'''
    random.seed(a=0)
    g = Game(5, 5, 2)
    assert(g.mines == {(0, 2), (3, 3)})
    assert(not g.guess((0, 2)))
    assert(g.mines == {(3, 3), (4, 3)})
    assert(g.guess((3, 3)))
    assert(g.mines == {(3, 3), (4, 3)})


def test_same_guess_twice():
    random.seed(a=0)
    g = Game(2, 2, 4)
    assert(g.mines == {(0, 0), (0, 1), (1, 0), (1, 1)})
    assert(not g.guess((0, 4)))
    assert(g.guess((0, 4)) is None)


def test_full_mines():
    random.seed(a=0)
    g = Game(2, 2, 4)
    assert(g.mines == {(0, 0), (0, 1), (1, 0), (1, 1)})


def test_game_repr():
    random.seed(a=0)
    g = Game(4, 4, 1)
    assert(not g.guess((1, 1)))
    assert(repr(g))


def test_is_won():
    random.seed(a=1)
    g = Game(4, 4, 2)
    assert(g.mines == {(2, 0), (1, 0)})
    assert(not g.is_won())
    assert(not g.guess((3, 3)))
    assert(not g.is_won())
    assert(not g.guess((3, 0)))
    assert(not g.is_won())
    assert(not g.guess((0, 0)))
    assert(g.is_won())


def test_view():
    random.seed(a=1)
    g = Game(2, 2, 1)
    g.guess((0, 0))
    assert(np.all(g.view() == np.array([[1, 9], [9, 9]])))


def test_format_move():
    random.seed(a=1)
    g = Game(2, 2, 1)
    g.guess((0, 0))
    assert(format_move(g.view(), g.mines, (0, 0), basic_style) == 'o \n  ')
    g.guess((1, 1))
    assert(format_move(g.view(), g.mines, (1, 1), basic_style) == '1 \n o')
    g.guess((1, 0))
    assert(format_move(g.view(), g.mines, (1, 0), basic_style) == '1 \nx1')


if __name__ == '__main__':
    test_simple_game()
    test_no_bad_luck()
    test_same_guess_twice()
    test_full_mines()
    test_game_repr()
    test_is_won()
    test_view()
    test_format_move()
