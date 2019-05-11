import glob
import datetime
import os
import pickle
from player import Player

def get_subdir(p):
  return os.path.join('data', str(p.height) + 'x' + str(p.width))

def dump_data(p):
  fname = str(datetime.datetime.now()) + ".pickle"
  subdir = get_subdir(p)
  if not os.path.exists(subdir):
    os.mkdir(subdir)
  with open(os.path.join(subdir, fname), 'wb') as f:
    pickle.dump(p.data, f)
  p.data = []


def load_data(p):
  p.data = []
  subdir = get_subdir(p)
  for fname in glob.glob(os.path.join(subdir, "*.pickle")):
    print("loading ", fname)
    with open(fname, "rb") as f:
      p.data.extend(pickle.load(f))


p = Player(5, 5, 3)

print("Training...")
load_data(p)
p.train()
p.data = []

for _ in range(100):
  p.play(5000)
  p.train(factor=2)
  p.play(1, debug=True)
  dump_data(p)
