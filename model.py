import random
import numpy as np
import tensorflow as tf
from config import LEARNING_RATE, L2_REGULARISATION, PRINT_ITERATIONS

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
        self.x = tf.placeholder(tf.float32, [None, 10 * n])

        W1 = tf.get_variable('W1', [10 * n, 20 * n])
        b1 = tf.get_variable('b1', [20 * n])
        z1 = tf.matmul(self.x, W1) + b1
        a1 = tf.nn.relu(z1)

        W2 = tf.get_variable('W2', [20 * n, 10 * n])
        b2 = tf.get_variable('b2', [10 * n])
        z2 = tf.matmul(a1, W2) + b2
        a2 = tf.nn.relu(z2)

        W3 = tf.get_variable('W3', [10 * n, 5 * n])
        b3 = tf.get_variable('b3', [5 * n])
        z3 = tf.matmul(a2, W3) + b3
        a3 = tf.nn.relu(z3)

        W4 = tf.get_variable('W4', [5 * n, n])
        b4 = tf.get_variable('b4', [n])
        z4 = tf.matmul(a3, W4) + b4

        self.p = tf.nn.sigmoid(z4)
        self.p_ = tf.placeholder(tf.float32, [None, n])

        loss_p = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.p_, logits=z4), reduction_indices=[1]))
        regulariser = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4)

        self.loss = loss_p + beta * regulariser
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

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
        self.measure_batch(*test_data).print_testing()

        for epoch in range(epochs):
            print("Epoch %3i" % epoch)
            random.shuffle(examples)
            results = []
            for idx in range(0, len(examples), batch_size):
                batch = examples[idx:idx+batch_size]
                grids, ps = list(zip(*batch))

                result = self.train_batch(grids, ps)
                results.append(result)

                if idx % PRINT_ITERATIONS < batch_size:
                    ModelBatchResults.combine(results).print_training(idx+batch_size, len(examples))
                    results = []
            if results:
                ModelBatchResults.combine(results).print_training(len(examples), len(examples))
                results = []
            self.measure_batch(*test_data).print_testing()

    def train_batch(self, grids, ps):
        feed_dict = {self.x: grids, self.p_: ps}
        self.sess.run(self.train_step, feed_dict=feed_dict)
        loss = self.sess.run([self.loss], feed_dict=feed_dict)[0]
        pred = self.sess.run([self.p], feed_dict=feed_dict)[0]
        return ModelBatchResults.from_prediction(loss, pred, ps)

    def measure_batch(self, grids, ps):
        feed_dict = {self.x: grids, self.p_: ps}
        loss = self.sess.run([self.loss], feed_dict=feed_dict)[0]
        pred = self.sess.run([self.p], feed_dict=feed_dict)[0]
        return ModelBatchResults.from_prediction(loss, pred, ps)

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

    def save(self, path):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, path)
        print("Model saved to %s" % save_path)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        print("Model restored")
