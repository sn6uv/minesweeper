import random

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
  None: ' ',
}


def format_move(game, pos, style=None):
  if style is None:
    style = basic_style
  view = game.view()
  result = [[style[v] for v in row] for row in view]
  if pos is not None:
    i, j = pos
    result[i][j] = 'x' if pos in game.mines else 'o'
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
    return format_move(self, None)

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

  def neighbors(self, pos):
    i, j = pos
    for ii in range(max(i-1, 0), min(i+2, self.height)):
      for jj in range(max(j-1, 0), min(j+2, self.width)):
        yield (ii, jj)

  def count_nearby_mines(self, pos):
    result = 0
    for n in self.neighbors(pos):
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
    for n in self.neighbors(pos):
      self.spread(n)

  def is_won(self):
    return len(self.guessed) + len(self.mines) == self.height * self.width

  def view(self):
    '''machine readable representation of what's seen'''
    rows = []
    for i in range(self.height):
      row = []
      for j in range(self.width):
        if (i,j) in self.guessed:
          row.append(self.count_nearby_mines((i,j)))
        else:
          row.append(None)
      rows.append(row)
    return rows
