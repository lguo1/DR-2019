Aimport tensorflow as tf
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

# fold 0, check 1, bet 2.
class node:
    def __init__(self, name):
        self.name = name

    def set_fold(self, child):
        self.fold = child
        child.before = [self, 0]

    def set_check(self, child):
        self.check = child
        child.before = [self, 1]

    def set_bet(self, child):
        self.bet = child
        child.before = [self, 2]

    def deal(self):
         return self.neighbors[np.random.choice(6, 1)]

    def take(self, action):
        return self.neighbors[action]

class game:
    def __init__(self):
        self.perms = ["01", "02", "10", "12", "20", "21"]
        self.terminal = "EIJGH"
        tree = {}
        for perm in self.perms:
            for g_node in "JIHGFEDCB":
                key = g_node + perm
                node = node(key)
                tree[key] = node
                if g_node == "B":
                    node.set_check(tree["C"+perm])
                    node.set_bet(tree["D"+perm])
                    node.neighbors = [None, node.check, node.bet]
                    node.P = 0
                    node.A = [1,2]
                    node.info = ([0,0,0],[0,0,0])
                elif g_node == "C":
                    node.set_check(tree["E"+perm])
                    node.set_bet(tree["F"+perm])
                    node.neighbors = [None, node.check, node.bet]
                    node.P = 1
                    node.A = [1,2]
                    node.info = ([1,0,0],[1,0,0])
                elif g_node == "D":
                    node.set_fold(tree["G"+perm])
                    node.set_bet(tree["H"+perm])
                    node.neighbors = [node.fold, None, node.bet]
                    node.P = 1
                    node.A = [0,2]
                    node.info = ([2,0,0],[1,0,0])
                elif g_node == "F":
                    node.set_fold(tree["I"+perm])
                    node.set_bet(tree["J"+perm])
                    node.neighbors = [node.fold, None, node.bet]
                    node.P = 0
                    node.A = [0,2]
                    node.info = ([1,2,0],[1,1,0])
                elif g_node == "E":
                    util = [-1,-1]
                    util[np.argmax(perm)] = 1
                    node.U = util
                elif g_node == "I":
                    node.U = [-1,1]
                elif g_node == "J":
                    util = [-2,-2]
                    util[np.argmax(perm)] = 2
                    node.U = util
                elif g_node == "G":
                    node.U = [1,-1]
                elif g_node == "H":
                    util = [-2,-2]
                    util[np.argmax(perm)] = 2
                    node.U = util
                else:
                    raise
        A = node("A")
        A.neighbors = [A.IJ, A.IK, A.JI, A.JK, A.KI, A.KJ]
        A.svalue
        for i in range(6):
            A.neighbors[i] = tree["B"+self.perms[i]]
        tree["A"] = A
        self.root = A
        self.tree = tree
        self.i_set = [[["01", "02"], ["10", "12"], ["20", "21"]], [["10", "20"], ["01", "21"], ["02", "12"]]]

    def i_set(self, p, perm):
        return self.i_set[p, perm[p+1]]

    def collect_samples(self, node, p, p_not, M_r, B_vp, B_s):
        if node.name[0] in self.terminal:
            return node.U[p]
        elif node.P == p:
            I = node.I(p)
            A = node.A
            sigma = calculate_strategy(I, A, M_r[p])
            v_a = np.zeros(3)
            for a in A:
                v_a[a] = self.collect_samples(node.take(a), p, p_not, M_r, B_vp, B_s)
            v_s = np.dot(v_a, sigma)
            d = v_a - v_s
            B_vp.add(I, d)
            return v_s
        elif node.P == p_not:
            I = node.I(p)
            A = node.A
            sigma = calculate_strategy(I, A, M_r[p_not])
            B_s.add(I, sigma)
            try:
                a = np.random.choice(3, p=sigma)
            except ValueError:
                a = np.random.choice(3, p=sigma/sigma.sum())
            return self.collect_samples(node.take(a), p, p_not, M_r, B_vp, B_s)
        else:
            return self.collect_samples(node.deal(), p, p_not, M_r, B_vp, B_s)

    def forward_update(self, M_r):
        queue = Queue()
        queue.enqueue(self.root)
        while not queue.is_empty():
            node = queue.dequeue()
            p = node.P
            A = node.A
            I = node.I(p)
            sigma = calculate_strategy(I, A, M_r[p])
            for a in A:
                neighbor = node.neighbors[a]
                if p == 0:
                    neighbor.prob[0] = sigma[a]*node.prob[0]
                    neighbor.prob[1] = node.prob[1]
                elif p == 1:
                    neighbor.prob[0] = node.prob[0]
                    neighbor.prob[1] = sigma[a]*node.prob[1]
                else:
                    neighbor.prob[0] = sigma[a]*node.prob[0]
                    neighbor.prob[1] = sigma[a]*node.prob[1]

    def backward_update(self):
        for g_node in "IJEFGHCDBA":
            for perm in self.perms:
                key = g_node + perm
                node = self.tree[key]
                if g_node in self.terminal:
                    node.svalue = node.U
                elif node.name == "A":
                    expected = [0,0]
                    for a in node.A:
                        neighbor = node.neighbors[a]
                        expected[1] += neighbor.prob[0]*neighbor.svalue[1]
                        expected[0] += neighbor.prob[1]*neighbor.svalue[0]
                    node.svalue = expected

                elif node.P == 0:
                    # exploit p0
                    expected = 0
                    for a in node.A:
                        neighbor = node.neighbors[a]
                        expected += neighbor.prob[0]*neighbor.svalue[1]
                    node.svalue[1] = expected
                    # exploit p1
                    sigma = np.zeros(3)
                    avalue = np.zeros((2,3))
                    n_set = []
                    for i in range(2):
                        i_node = self.tree[g_node + self.i_set(0, perm)[i]]
                        n_set.append(i_node)
                        for a in i_node.A:
                            avalue[i,a] = i_node.neighbors[a].svalue[0]
                        sigma[np.argmax(avalue[i])] += i_node.prob[1]
                    n_set[0].svalue[0] = np.sum(sigma*avalue)/np.sum(sigma)
                    n_set[1].svalue[0] = n_set[0].svalue[0]

                elif node.P == 1:
                    # exploit p0
                    sigma = np.zeros(3)
                    avalue = np.zeros((2,3))
                    n_set = []
                    for i in range(2):
                        i_node = self.tree[g_node + self.i_set(1, perm)[i]]
                        n_set.append(i_node)
                        for a in i_node.A:
                            avalue[i,a] = i_node.neighbors[a].svalue[1]
                        sigma[np.argmax(avalue[i])] += i_node.prob[0]
                    n_set[0].svalue[1] = np.sum(sigma*avalue)/np.sum(sigma)
                    n_set[1].svalue[1] = n_set[0].svalue[1]
                    # exploit p1
                    expected = 0
                    for a in node.A:
                        neighbor = node.neighbors[a]
                        expected += neighbor.prob[1]*neighbor.svalue[0]
                    node.svalue[0] = expected
                else:
                    raise
                return self.root.svalue

def connect(input, weights, biases, activations):
    layer = input
    for W, b, activation in zip(weights, biases, activations):
        layer = tf.matmul(layer, W) + b
        if activation is not None:
            layer = activation(layer)
    return layers
