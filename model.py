import random
import numpy as np
import tensorflow.compat.v1 as tf
from config import LEARNING_RATE, L2_REGULARISATION, PRINT_ITERATIONS
from utils import model_path

tf.disable_v2_behavior()

class ModelBatchResults:
    '''
    The result of training the model on one batch.
    '''
    def __init__(self, loss, precision, recall, accuracy):
        self.loss = loss
        self.precision = precision
        self.recall = recall
        self.accuracy = accuracy

    @staticmethod
    def from_prediction(loss, prediction, actual_mines):
        actual_mines = np.array(actual_mines)
        result = ModelBatchResults(None, None, None, None)
        result.loss = loss
        predicted_mines = prediction > 0.5
        true_pos = np.sum(np.logical_and((predicted_mines == actual_mines), (predicted_mines == 1)))
        result.precision = true_pos / np.sum(predicted_mines == 1)
        result.recall = true_pos / np.sum(actual_mines == 1)
        result.accuracy = np.sum(predicted_mines == actual_mines) / prediction.size
        return result

    def print_training(self, iteration, total_iterations):
        print("%5.1f%% iteration: %7i loss: %6.3f precision: %5.3f recall: %5.3f accuracy: %5.3f"
          % (100.0 * iteration / total_iterations, iteration, self.loss, self.precision, self.recall, self.accuracy))

    def print_testing(self):
        print("Testing:                  loss: %6.3f precision: %5.3f recall: %5.3f accuracy: %5.3f"
          % (self.loss, self.precision, self.recall, self.accuracy))

    @staticmethod
    def combine(results):
        n = len(results)
        result = ModelBatchResults(None, None, None, None)
        result.loss = sum(r.loss for r in results) / n
        result.precision = sum(r.precision for r in results) / n
        result.recall = sum(r.recall for r in results) / n
        result.accuracy = sum(r.accuracy for r in results) / n
        return result


class Model:
    '''
    A model for predicting the location of mines in a minesweeper game.
    '''

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.build_model()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def build_model(self, learning_rate=LEARNING_RATE, beta=L2_REGULARISATION):
        n = self.height * self.width
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(self.height, self.width, 10)))
        self.model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        self.model.add(keras.layers.Flatten())
        reg = keras.regularizers.l2(beta)
        self.model.add(tf.keras.layers.Dense(5 * n, activation='relu', kernel_regularizer=reg))
        self.model.add(tf.keras.layers.Dense(n, activation='sigmoid', kernel_regularizer=reg))
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
        self.model.fit(np.array(train_data[0]), np.array(train_data[1]), batch_size=batch_size, epochs=epochs)

        _, acc = self.model.evaluate(np.array(test_data[0]), np.array(test_data[1]))

    def predict(self, grid):
        """Evaluates the model to predict an output.

        Args:
            grid: a game state as a height*width*10 vector.

        Returns:
            p: probability distribution over moves.
        """
        grid = grid[np.newaxis, :]
        p = self.sess.run([self.p], feed_dict={self.x: grid})
        return p[0]

    def save(self, name):
        path = model_path(name)
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, path)
        print("Model saved to %s" % save_path)

    def restore(self, name):
        path = model_path(name)
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        print("Model restored")
