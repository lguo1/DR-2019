import tensorflow as tf
import numpy as np
from operator import itemgetter

class buffer:
    def __init__(self):
        self.list = [[],[],[],[]]
        self.size = 0
        self.count = 0

    def add(self, input, output):
        self.list[0].append(input[0])
        self.list[1].append(input[1])
        self.list[2].append(input[2])
        self.list[3].append(output)
        self.size += 1
        self.count += 1
        return self.list

    def set(self):
        self.count = 0

    def sample(self, indices):
        return (itemgetter(*indices)(self.list[0]), itemgetter(*indices)(self.list[1]), itemgetter(*indices)(self.list[2]), itemgetter(*indices)(self.list[3]))

def connect(input, weights, biases, activations):
    layer = input
    for W, b, activation in zip(weights, biases, activations):
        layer = tf.matmul(layer, W) + b
        if activation is not None:
            layer = activation(layer)
    return layer

class model:
    def __init__(self, name, softmax=False):
        self.name = name
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope(name):
                self.output_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3])

                b1d = tf.get_variable(name='b1d', shape=[16], initializer=tf.constant_initializer(0.))
                b2d = tf.get_variable(name='b2d', shape=[16], initializer=tf.constant_initializer(0.))
                b3d = tf.get_variable(name='b3d', shape=[3], initializer=tf.constant_initializer(0.))
                W2d = tf.get_variable(name='W2d', shape=[16, 16], initializer=tf.contrib.layers.xavier_initializer())
                W3d = tf.get_variable(name='W3d', shape=[16, 3], initializer=tf.contrib.layers.xavier_initializer())

                self.input_pha = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                W0a = tf.get_variable(name='W0a', shape=[1, 32], initializer=tf.contrib.layers.xavier_initializer())
                W1a = tf.get_variable(name='W1a', shape=[32, 16], initializer=tf.contrib.layers.xavier_initializer())
                b0a = tf.get_variable(name='b0a', shape=[32], initializer=tf.constant_initializer(0.))
                activations = [tf.nn.relu, None]
                weights = [W0a, W1a]
                biases = [b0a, b1d]
                outputa = connect(self.input_pha, weights, biases, activations)

                self.input_phb = tf.placeholder(dtype=tf.float32, shape=[None, 3])
                W0b = tf.get_variable(name='W0b', shape=[3, 32], initializer=tf.contrib.layers.xavier_initializer())
                W1b = tf.get_variable(name='W1b', shape=[32, 16], initializer=tf.contrib.layers.xavier_initializer())
                b0b = tf.get_variable(name='b0b', shape=[32], initializer=tf.constant_initializer(0.))
                weights = [W0b, W1b]
                biases = [b0b, b1d]
                outputb = connect(self.input_phb, weights, biases, activations)

                self.input_phc = tf.placeholder(dtype=tf.float32, shape=[None, 3])
                W0c = tf.get_variable(name='W0c', shape=[3, 32], initializer=tf.contrib.layers.xavier_initializer())
                W1c = tf.get_variable(name='W1c', shape=[32, 16], initializer=tf.contrib.layers.xavier_initializer())
                b0c = tf.get_variable(name='b0c', shape=[32], initializer=tf.constant_initializer(0.))
                weights = [W0c, W1c]
                biases = [b0c, b1d]
                outputc = connect(self.input_phc, weights, biases, activations)

                inputd = outputa + outputb + outputc
                if softmax:
                    activations = [tf.nn.relu, tf.nn.softmax]
                else:
                    activations = [tf.nn.relu, None]
                weights = [W2d, W3d]
                biases = [b2d, b3d]
                self.output_pred = connect(inputd, weights, biases, activations)

                self.mse = tf.reduce_mean(0.5 * tf.square(self.output_pred - self.output_ph))
                self.opt = tf.train.AdamOptimizer().minimize(self.mse)

                self.sess = tf.Session()
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver()

    def predict(self, inputs):
        return self.sess.run(self.output_pred, feed_dict={self.input_pha: [inputs[0]], self.input_phb: [inputs[1]], self.input_phc: [inputs[2]]})

    def train(self, B, weights, N_train, N_batch, save=False):
        for training_step in range(N_train):
            weights = np.array(weights)
            sample = B.sample(np.random.choice(B.size, N_batch, p=weights/weights.sum()))
            _, mse_run = self.sess.run([self.opt, self.mse], feed_dict={self.input_pha: sample[0], self.input_phb: sample[1], self.input_phc: sample[2], self.output_ph: sample[3]})
            if training_step % 1000 == 0:
                print('%s: mse: %0.3f'%(self.name, mse_run))
            if save:
                self.saver.save(self.sess, './saves/%s_model.ckpt'%(self.name))

    def restore(self):
        self.saver.restore(self.sess, './saves/%s_model.ckpt'%(self.name))

# fold 0; check 1; bet 2.
class game:
    def __init__(self):
        self.all_perms = permutations(range(3),2)
        self.tree = {
        "B": ["", "C", "D"],
        "C": ["", "E", "F"],
        "D": ["G", "", "H"],
        "F": ["I", "", "J"]
        }
        self.before = {
        "I": "F",
        "J": "F",
        "E": "C",
        "F": "C",
        "G": "D",
        "H": "D",
        "C": "B",
        "D": "B"
        }
        self.info = {
        "B": [[0,0,0],[0,0,0]],
        "C": [[1,0,0],[1,0,0]],
        "D": [[2,0,0],[1,0,0]],
        "E": [[1,1,0],[1,1,0]],
        "F": [[1,2,0],[1,1,0]],
        "G": [[2,0,0],[1,1,0]],
        "H": [[2,2,0],[1,1,0]],
        "I": [[1,2,1],[1,1,1]],
        "J": [[1,2,2],[1,1,1]]
        }
        self.available = {
        "B": [1,2],
        "C": [1,2],
        "D": [0,2],
        "F": [0,2]
                        }
        self.layers = ["B","CD","EFGH","IJ"]
        self.terminal = "EIJGH"

    def deal(self):
        self.cards = np.random.choice(3,(2,1),replace=False)
        return self

    def util(self, node, p):
        if node == "E":
            return (p == np.argmax(self.cards))*2-1
        elif node == "I":
            return [-1,1][p]
        elif node == "J":
            return (p == np.argmax(self.cards))*4-2
        elif node == "G":
            return [1,-1][p]
        elif node == "H":
            return (p == np.argmax(self.cards))*4-2
        else:
            return None

    def take(self, node, action):
        return self.tree[node][action]

    def A(self, node):
        return self.available[node]

    def P(self, node):
        if node in ["C", "D"]:
            return 1
        elif node in ["B", "F"]:
            return 0
        else:
            return None

    def I(self, node, p):
        return (self.cards[p], *self.info[node])
