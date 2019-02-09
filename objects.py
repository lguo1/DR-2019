import tensorflow as tf
import numpy as np

class buffer:
    def __init__(self):
        self.list = [[],[],[],[]]
        self.size = 0
    def add(self, input, output):
        self.list[0].append(input[0])
        self.list[1].append(input[1])
        self.list[2].append(input[2])
        self.list[3].append(output)
        self.size += 1
        return self.list

    def sample(self, indices):
        return (self.list[0,indices], self.list[1,indices], self.list[2,indices], self.list[3,indices])

def connect(input, weights, biases, activations):
    layer = input
    for W, b, activation in zip(weights, biases, activations):
        layer = tf.matmul(layer, W) + b


class model:
    def __init__(self, name, softmax=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope(name):
                self.output_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3])

                b1_D = tf.get_variable(name='D/b1'%name, shape=[16], initializer=tf.constant_initializer(0.))

                self.input_ph_A = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                W0_A = tf.get_variable(name='A/W0'%name, shape=[1, 32], initializer=tf.contrib.layers.xavier_initializer())
                W1_A = tf.get_variable(name='A/W1'%name, shape=[32, 16], initializer=tf.contrib.layers.xavier_initializer())
                b0_A = tf.get_variable(name='A/b0'%name, shape=[32], initializer=tf.constant_initializer(0.))

                self.input_ph_B = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                W0_B = tf.get_variable(name='B/W0', shape=[1, 32], initializer=tf.contrib.layers.xavier_initializer())
                W1_B = tf.get_variable(name='B/W1', shape=[32, 16], initializer=tf.contrib.layers.xavier_initializer())
                b0_B = tf.get_variable(name='B/b0', shape=[32], initializer=tf.constant_initializer(0.))

                self.input_ph_C = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                W0_C = tf.get_variable(name='C/W0', shape=[1, 32], initializer=tf.contrib.layers.xavier_initializer())
                W1_C = tf.get_variable(name='C/W1', shape=[32, 16], initializer=tf.contrib.layers.xavier_initializer())
                b0_C = tf.get_variable(name='C/b0', shape=[32], initializer=tf.constant_initializer(0.))

                activations = [tf.nn.relu, None]
                weights = [W0_0, W0_1]
                biases = [b0_0, b0_1]
                layer0 = self.input_ph0
                layer0 = tf.nn.relu(tf.matmul(layer0, W0_0) + b0_0)
                layer0 = tf.matmul(layer0, W0_1) + b0_1

                self.input_ph1 = tf.placeholder(dtype=tf.float32, shape=[None, 3])
                W1_0 = tf.get_variable(name='W1_0', shape=[3, 32], initializer=tf.contrib.layers.xavier_initializer())
                W0_0 = tf.get_variable(name='W0_0', shape=[1, 32], initializer=tf.contrib.layers.xavier_initializer())
                W0_1 = tf.get_variable(name='W0_1', shape=[32, 16], initializer=tf.contrib.layers.xavier_initializer())
                b0_0 = tf.get_variable(name='b0_0', shape=[32], initializer=tf.constant_initializer(0.))
                b0_1 = tf.get_variable(name='b2_0', shape=[16], initializer=tf.constant_initializer(0.))
                b1_0 = b0_1
                layer1 = self.input_ph1
                layer1 = tf.layers.conv2d(layer1, 32, (3,2), activation=tf.nn.relu)
                layer1 = tf.layers.conv2d(layer1, 32, 1, activation=tf.nn.relu)
                layer1 = tf.contrib.layers.flatten(layer1)
                layer1 = tf.matmul(layer1, W1_0) + b1_0

                layer2 = tf.nn.relu(layer0 + layer1)
                W2_0 = tf.get_variable(name='W2_1', shape=[16, 3], initializer=tf.contrib.layers.xavier_initializer())
                b2_0 = tf.get_variable(name='b2_1', shape=[3], initializer=tf.constant_initializer(0.))
                if softmax:
                    self.output_pred = tf.nn.softmax(tf.matmul(layer2, W2_0) + b2_0)
                else:
                    self.output_pred = tf.matmul(layer2, W2_0) + b2_0
                init_graph = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(init_graph)

    def predict(self, inputs):
        return self.sess.run(self.output_pred, feed_dict={self.input_ph0: [inputs[0]], self.input_ph1: [inputs[1]]})

    def train(self, B, weights, N_train, N_batch):
        mse = tf.reduce_mean(0.5 * tf.square(self.output_pred - self.output_ph))
        opt = tf.train.AdamOptimizer().minimize(mse)
        saver = tf.train.Saver()
        for training_step in range(N_train):
            sample = B.sample(tf.contrib.training.weighted_resample(range(B.size), weights, N_batch/B.size))
            _, mse_run = self.sess.run([opt, mse], feed_dict={self.input_ph0: sample[0], self.input_ph1: sample[1], self.output_ph: sample[2]})
            if training_step % 1000 == 0:
                print('{0:04d} mse: {1:.3f}'.format(training_step, mse_run))
                saver.save(sess, '/saved/%s_%d.ckpt'%(self.name, training_step))



# fold 0; check 1; bet 2.
class node:
    def __init__(self):
        self.actions = np.zeros(3)
        self.prog = np.zeros(3)
        self.turns = -1
        self.is_terminal = False

    def deal(self):
        self.cards = np.random.choice(3,(2,1))
        self.turns += 1
        return self

    def place(self, action):
        self.actions[self.turns] = action
        self.prog[self.turns] = 1
        self.turns += 1
        return self

    def A(self):
        if self.turns == 0:
            return [1,2]
        elif self.turns == 1:
            if self.actions[0] == 1:
                return [1,2]
            else:
                return [0,2]
        else:
            return [0,2]

    def P(self):
        if self.turns == -1:
            return -1
        else:
            return (self.turns+1)%2

    def is_terminal(self):
        if self.prog[2]:
            return True
        elif not self.prog[1]:
            return False
        elif self.actions[1] != 2:
            return True
        elif self.actions[0] == 2:
            return True
        else:
            return False

    def util(self, p):
        thd = self.actions[2]
        if thd == 0:
            return [0,3][p]
        elif thd == 2:
            return (p == np.argmax(self.cards))*4
        else:
            sec = self.actions[1]
            if sec == 1:
                return (p == np.argmax(self.cards))*2
            elif sec == 2:
                return (p == np.argmax(self.cards))*4
            else:
                return [3,0][p]

    def I(self, p):
        return [self.cards[p], self.actions, self.prog]
