# minesweeper

Playing minesweeper with neural networks

## Approach

The AI player chooses the lowest probability location for its next guess. This
is a suboptimal strategy since a higher risk guess may reveal more information,
improving the chance of winning that game.

The goal is to minimise the product of turn probabilities. For example, an algorithm
that finishes the game in 10 moves, each with 1% failure, will win 0.99^10 = 90.4%
of games while an algorithm that finishes in 100 moves with 0.5% failure rate will
only win 0.995^100 = 60.6% of games.

## Implementation

### Libraries

#### player.py

Uses a neural net (NN) to predict where mines are going to be. The AI player
chooses the lowest probability location for its next guess.

#### model.py

The NN consists of 4 fully connected layers whose sizes are dynamic depending
on the game dimensions. A logistic loss function is used.

#### game.py

This variant of minesweeper can be any size. The first guess is guarenteed to
be safe by moving a mine if the first guess is unlucky.

### Binaries

#### play.py

Plays trial games to generate training data.

#### train.py

Trains on all saved training data.

## Results

### 4x4 with 2 mines

Wins >95% of games (<1 min of training).

### 5x5 with 3 mines

Wins >60% of games (2 mins of training).

### 9x9 with 10 mines (Beginner)

Wins 41% of games (after several days training on CPU).

![gallery](https://raw.githubusercontent.com/sn6uv/minesweeper/master/results/gallery.png)
