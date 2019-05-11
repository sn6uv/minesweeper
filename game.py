import random

def format_move(game, pos):
  lines = []
  for i in range(game.height):
    line = []
    for j in range(game.width):
      if (i, j) == pos:
        if pos in game.mines:
          line.append('x')
        else:
          line.append('o')
      elif (i,j) in game.guessed:
        line.append(str(game.count_nearby_mines((i,j))))
      else:
        line.append(' ')
    lines.append(line)
  return '\n'.join(''.join(line) for line in lines)


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
