import random
import numpy as np
import tensorflow as tf
from config import LEARNING_RATE, L2_REGULARISATION, PRINT_ITERATIONS
from utils import model_path
from tensorflow  import keras


class Model:
    '''
    A model for predicting the location of mines in a minesweeper game.
    '''

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.build_model()

    def build_model(self, learning_rate=LEARNING_RATE, beta=L2_REGULARISATION):
        n = self.height * self.width
        inputs = tf.keras.Input(shape=(10*n,))
        x = tf.keras.layers.Dense(20 * n, activation='relu')(inputs)
        x = tf.keras.layers.Dense(10 * n, activation='relu')(x)
        x = tf.keras.layers.Dense(5 * n, activation='relu')(x)
        outputs = tf.keras.layers.Dense(n, activation='sigmoid')(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, examples, batch_size=5000, epochs=1):
        """Trains the model on examples.

        Args:
            examples: list of examples, each example is of the form (grid, p).
        """

        # Split a fraction of data for testing. Since data comes from playing
        # games, if we take a random subset for testing then it's correlated
        # with the training data. Taking a slice off the end mitigates this.
        num_test_examples = int(len(examples) * 0.01)
        examples, test_examples = examples[:-num_test_examples], examples[-num_test_examples:]
        test_data = list(zip(*test_examples))
        train_data = list(zip(*examples))
        self.model.fit(np.matrix(train_data[0]), np.matrix(train_data[1]), batch_size=batch_size, epochs=epochs)

        _, acc = self.model.evaluate(np.matrix(test_data[0]), np.matrix(test_data[1]))

    def predict(self, grid):
        """Evaluates the model to predict an output.

        Args:
            grid: a game state as a height*width*10 vector.

        Returns:
            p: probability distribution over moves.
        """
        grid = grid[np.newaxis, :]
        # Adding batch_size increases performance significantly.
        # https://stackoverflow.com/questions/48796619/why-is-tf-keras-inference-way-slower-than-numpy-operations
        return self.model.predict(grid, batch_size=1)

    def save(self, name):
        path = model_path(name)
        self.model.save(path)
        print("Model saved to %s" % path)

    def restore(self, name):
        path = model_path(name)
        self.model = tf.keras.load_model(path)
        print("Model restored from %s" % path)
