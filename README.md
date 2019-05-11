# minesweeper

Playing minesweeper with neural networks

## Implementation

### player.py

Uses a neural net (NN) to predict where mines are going to be. The AI player
chooses the lowest probability location for its next guess.

### model.py

The NN consists of 4 fully connected layers whose sizes are dynamic depending
on the game dimensions. A logistic loss function is used.

### game.py

This variant of minesweeper can be any size. The first guess is guarenteed to
be safe by moving a mine if the first guess is unlucky.

## Results

### 4x2 with 2 mines

Wins >95% of games (<1 min of training).

### 5x5 with 3 mines

Wins >60% of games (2 mins of training).

### 9x9 with 10 mines (Beginner)

TODO
