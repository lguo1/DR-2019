import tensorflow as tf
import numpy as np
from operator import itemgetter
from queue import Queue
from itertools import permutations
import pickle
import sys # remove later

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
    def __init__(self, name, sigmoid=False):
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
                if sigmoid:
                    activations = [tf.nn.relu, tf.nn.sigmoid]
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
        print("     %s's mse: %0.3f"%(self.name, mse_run))
        if save:
            self.saver.save(self.sess, './saves/%s_model.ckpt'%(self.name))

    def restore(self):
        self.saver.restore(self.sess, './saves/%s_model.ckpt'%(self.name))

# fold 0, check 1, bet 2.
class Node:
    def __init__(self, name):
        self.name = name
        self.A = None
        self.P = None
        self.prob = [None, None]
        self.value = [None, None]
        self.n_perm = None

    def set_fold(self, child):
        self.fold = child
        child.before = [self, 0]
        return child

    def set_check(self, child):
        self.check = child
        child.before = [self, 1]
        return child

    def set_bet(self, child):
        self.bet = child
        child.before = [self, 2]
        return child

    def deal(self):
        return self.neighbors[np.random.choice(6)]

    def take(self, action):
        return self.neighbors[action]

    def I(self, p):
        return ([self.n_perm[p]], *self.info)

    def U(self, p):
        return self.value[p]

    def rootify(self, tree):
        self.IJ = tree["B01"]
        self.IK = tree["B02"]
        self.JI = tree["B10"]
        self.JK = tree["B12"]
        self.KI = tree["B21"]
        self.KJ = tree["B20"]
        self.neighbors = [self.IJ, self.IK, self.JI, self.JK, self.KI, self.KJ]
        self.A = list(range(6))
        self.prob = [1,1]
        tree[self.name] = self
        return self

class Game:
    def __init__(self):
        self.perms = ["01", "02", "10", "12", "20", "21"]
        self.n_perms = list(permutations(range(3), 2))
        self.terminal = "EIJGH"
        tree = {}
        for i in range(6):
            for g_node in "JIHGFEDCB":
                perm = self.perms[i]
                n_perm = self.n_perms[i]
                key = g_node + self.perms[i]
                node = Node(key)
                node.n_perm = n_perm
                tree[key] = node
                if g_node == "B":
                    node.neighbors = [None, node.set_check(tree["C"+perm]), node.set_bet(tree["D"+perm])]
                    node.P = 0
                    node.A = [1,2]
                    node.info = ([0,0,0],[0,0,0])
                elif g_node == "C":
                    node.neighbors = [None, node.set_check(tree["E"+perm]), node.set_bet(tree["F"+perm])]
                    node.P = 1
                    node.A = [1,2]
                    node.info = ([1,0,0],[1,0,0])
                elif g_node == "D":
                    node.neighbors = [node.set_fold(tree["G"+perm]), None, node.set_bet(tree["H"+perm])]
                    node.P = 1
                    node.A = [0,2]
                    node.info = ([2,0,0],[1,0,0])
                elif g_node == "F":
                    node.neighbors = [node.set_fold(tree["I"+perm]), None, node.set_bet(tree["J"+perm])]
                    node.P = 0
                    node.A = [0,2]
                    node.info = ([1,2,0],[1,1,0])
                elif g_node == "E":
                    util = [-1,-1]
                    util[np.argmax(n_perm)] = 1
                    node.value = util
                elif g_node == "I":
                    node.value = [-1,1]
                elif g_node == "J":
                    util = [-2,-2]
                    util[np.argmax(n_perm)] = 2
                    node.value = util
                elif g_node == "G":
                    node.value = [1,-1]
                elif g_node == "H":
                    util = [-2,-2]
                    util[np.argmax(n_perm)] = 2
                    node.value = util
                else:
                    raise
        A = Node("A")
        self.root = A.rootify(tree)
        self.tree = tree
        self.i_set = [[["01", "02"], ["10", "12"], ["20", "21"]], [["10", "20"], ["01", "21"], ["02", "12"]]]

    def i_perm(self, perm, p):
        return self.i_set[p][int(perm[p])]

    def collect_samples(self, node, p, p_not, M_r, B_vp, B_s):
        if node.name[0] in self.terminal:
            return node.U(p)
        elif node.P == p:
            I = node.I(p)
            A = node.A
            sigma = calculate_strategy(I, A, M_r[p])
            v_a = np.full(3, -2)
            for a in A:
                v_a[a] = self.collect_samples(node.take(a), p, p_not, M_r, B_vp, B_s)
            v_s = np.dot(v_a, sigma)
            d = normalize(v_a - v_s)
            if node.name == "D01":
                print(">>>>>>>>")
                print(node.name)
                print("v_s",v_s)
                print("v_a",v_a)
                print("t_d", d)
                print("<<<<<<<<")
            B_vp.add(I, d)
            return v_s
        elif node.P == p_not:
            I = node.I(p_not)
            A = node.A
            sigma = calculate_strategy(I, A, M_r[p_not])
            B_s.add(I, sigma)
            a = np.random.choice(3, p=sigma)
            '''
            if node.name == "F20":
                print(node.name)
                print("sigma", sigma)
                print("a", a)
                print("____")
            '''
            return self.collect_samples(node.take(a), p, p_not, M_r, B_vp, B_s)
        else:
            return self.collect_samples(node.deal(), p, p_not, M_r, B_vp, B_s)

    def forward_update(self, model, t):
        dic = {}
        queue = Queue()
        queue.put(self.root)
        while not queue.empty():
            node = queue.get()
            A = node.A
            p = node.P
            if node.name == "A":
                sigma = np.full(6, 1/6)
            else:
                I = node.I(p)
                sigma = calculate_strategy(I, A, model)
            dic[node.name] = sigma
            for a in A:
                neighbor = node.neighbors[a]
                if neighbor.name[0] in "BCDF":
                    queue.put(neighbor)
                if p == 0:
                    neighbor.prob[0] = sigma[a]*node.prob[0]
                    neighbor.prob[1] = node.prob[1]
                elif p == 1:
                    neighbor.prob[0] = node.prob[0]
                    neighbor.prob[1] = sigma[a]*node.prob[1]
                else:
                    neighbor.prob[0] = sigma[a]*node.prob[0]
                    neighbor.prob[1] = sigma[a]*node.prob[1]
        with open('./sigmas/sigma-%d.pkl'%t, 'wb') as output:
            pickle.dump(dic, output, pickle.HIGHEST_PROTOCOL)

    def backward_update(self):
        for g_node in "FCDB":
            for perm in self.perms:
                key = g_node + perm
                node = self.tree[key]
                if node.P == 0:
                    # exploit p0
                    expected = 0
                    for a in node.A:
                        neighbor = node.neighbors[a]
                        expected += neighbor.prob[0]*neighbor.value[1]
                    node.value[1] = expected
                    # exploit p1
                    a_v = np.zeros((2,2))
                    n_set = []
                    norm = 0
                    for i in range(2):
                        i_node = self.tree[g_node + self.i_perm(perm, 0)[i]]
                        n_set.append(i_node)
                        norm += i_node.prob[1]
                        for j in range(2):
                            a_v[i,j] = i_node.neighbors[i_node.A[j]].value[0]*i_node.prob[1]
                    a_v = np.sum(a_v, axis = 0)
                    n_set[0].value[0] = np.max(a_v)/norm
                    n_set[1].value[0] = n_set[0].value[0]
                else:
                    # exploit p1
                    expected = 0
                    for a in node.A:
                        neighbor = node.neighbors[a]
                        expected += neighbor.prob[1]*neighbor.value[0]
                    node.value[0] = expected
                    # exploit p0
                    a_v = np.zeros((2,2))
                    n_set = []
                    norm = 0
                    for i in range(2):
                        i_node = self.tree[g_node + self.i_perm(perm, 1)[i]]
                        n_set.append(i_node)
                        norm += i_node.prob[0]
                        for j in range(2):
                            a_v[i,j] = i_node.neighbors[i_node.A[j]].value[1]*i_node.prob[0]
                    a_v = np.sum(a_v, axis = 0)
                    n_set[0].value[1] = np.max(a_v)/norm
                    n_set[1].value[1] = n_set[0].value[1]

        expected = [0,0]
        for neighbor in self.root.neighbors:
            expected[1] += neighbor.prob[0]*neighbor.value[1]
            expected[0] += neighbor.prob[1]*neighbor.value[0]
        self.root.value = expected
        return self.root.value

def connect(input, weights, biases, activations):
    layer = input
    for W, b, activation in zip(weights, biases, activations):
        layer = tf.matmul(layer, W) + b
        if activation is not None:
            layer = activation(layer)
    return layer

def calculate_strategy(I, A, model):
    sigma = np.zeros(3)
    d = model.predict(I)[0, A]
    d_plus = np.clip(d, 0, None)
    if d_plus.sum() > 0:
        sigma[A] = d_plus/np.sum(d_plus)
        if np.sum(sigma)!= 1:
            return sigma/np.sum(sigma)
        return sigma
    else:
        sigma[A[np.argmax(d)]] = 1
        return sigma

def normalize(x):
    return np.exp(x) / np.exp(x).sum()
