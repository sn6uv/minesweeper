import random
import numpy as np
from functools import lru_cache


basic_style = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: ' ',
    'o': 'o',
    'x': 'x',
}, lambda x, s:  s

default_style = basic_style

try:
    from sty import fg, bg
    sty_style = {
        0: bg.white + ' ' + bg.rs,
        1: bg.white + fg.blue + '1' + fg.rs + bg.rs,
        2: bg.white + fg.green + '2' + fg.rs + bg.rs,
        3: bg.white + fg.red + '3' + fg.rs + bg.rs,
        4: bg.white + fg.da_blue + '4' + fg.rs + bg.rs,
        5: bg.white + fg.da_red + '5' + fg.rs + bg.rs,
        6: bg.white + fg.da_green + '6' + fg.rs + bg.rs,
        7: bg.white + fg.magenta + '7' + fg.rs + bg.rs,
        8: bg.white + fg.black + '8' + fg.rs + bg.rs,
        9: ' ',
        'o': fg.black + 'o' + fg.rs,
        'x': fg.black + 'x' + fg.rs,
    }, lambda x, s: bg(int(x * 255.0), (255 - (int(x * 255.0))), 0) + s + bg.rs
    default_style = sty_style
except ImportError:
    pass


def neighbors_do(pos, height, width):
    '''Find the neighbors of a given position in a grid.'''
    i, j = pos
    for ii in range(max(i-1, 0), min(i+2, height)):
        for jj in range(max(j-1, 0), min(j+2, width)):
            yield (ii, jj)


@lru_cache(maxsize=81)
def neighbors(pos, height, width):
    return list(neighbors_do(pos, height, width))


def format_move(view, mines, pos, style=None, risk_matrix=None):
    if style is None:
        style = default_style
    result = [[style[0][v] for v in row] for row in view]
    if pos is not None:
        if pos in mines:
            for (i, j) in mines:
                result[i][j] = style[0]['x']
            result[pos[0]][pos[1]] = style[0]['o']
        else:
          i, j = pos
          result[i][j] = style[0]['o']
    if risk_matrix is not None:
        for i, row in enumerate(view):
            for j, v in enumerate(row):
                if v == 9:
                    r = risk_matrix[i][j]
                    result[i][j] = style[1](r, result[i][j])
    return '\n'.join(''.join(row) for row in result)


class Game:
    def __init__(self, height, width, mines):
        self.height = height
        self.width = width
        self.guessed = set()
        self.mines = set()
        self.add_mines(mines)

    # Adds mines randomly.
    def add_mines(self, mines):
        for _ in range(mines):
            while True:
                i = random.randint(0, self.height-1)
                j = random.randint(0, self.width-1)
                pos = (i, j)
                if pos in self.mines:
                    continue
                self.mines.add(pos)
                break

    def __repr__(self):
        return format_move(self.view(), self.mines, None)

    # Returns True if a mine was hit.
    def guess(self, pos):
        if not self.guessed and pos in self.mines:
            # If the first guess was unlucky then move the mine.
            while pos in self.mines:
                self.mines.remove(pos)
                self.add_mines(1)
        if pos in self.mines:
            return True
        if pos in self.guessed:
            return None
        self.spread(pos)
        return False


    def count_nearby_mines(self, pos):
        result = 0
        for n in neighbors(pos, self.height, self.width):
            if n in self.mines:
                result += 1
        return result

    def spread(self, pos):
        '''spreads a guess out'''
        if pos in self.guessed:
            return
        self.guessed.add(pos)
        if self.count_nearby_mines(pos) > 0:
            return
        for n in neighbors(pos, self.height, self.width):
            self.spread(n)

    def is_won(self):
        return len(self.guessed) + len(self.mines) == self.height * self.width

    def view(self):
        '''machine readable representation of what's seen'''
        result = np.zeros((self.height, self.width), dtype=np.int8) + 9
        for (i, j) in self.guessed:
            result[i, j] = self.count_nearby_mines((i, j))
        return result
