import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.build_model()
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def build_model(self, learning_rate=0.001, beta=0.01):
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
        losses = []
        for idx in range(0, len(examples) * epochs, batch_size):
            sample_ids = np.random.randint(len(examples), size=batch_size)
            grids, ps = list(zip(*[examples[i] for i in sample_ids]))

            feed_dict = {self.x: grids, self.p_: ps}
            self.sess.run(self.train_step, feed_dict=feed_dict)
            loss = self.sess.run([self.loss], feed_dict=feed_dict)
            losses.append(loss[0])
            if idx % 5000 < batch_size:
                print("loss at iteration %s is %s" % (idx, sum(losses) / len(losses)))
                losses = []

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
