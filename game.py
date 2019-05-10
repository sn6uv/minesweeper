import random

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
    lines = []
    for i in range(self.height):
      line = []
      for j in range(self.width):
        if (i,j) in self.guessed:
          line.append(str(self.count_nearby_mines((i,j))))
        else:
          line.append(' ')
      lines.append(line)
    return '\n'.join(''.join(line) for line in lines)

  # Returns True if a mine was hit.
  def guess(self, pos):
    if pos in self.mines:
      return True
    assert(pos not in self.guessed)
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


def BeginnerGame():
  return Game(9, 9, 10)


def IntermediateGame():
  return Game(16, 16, 40)


def AdvancedGame():
  return Game(24, 24, 99)
