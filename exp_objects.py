import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from operator import itemgetter
from queue import Queue
from itertools import permutations
import pickle

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

    def set(self):
        self.count = 0

    def sample(self, indices):
        return (itemgetter(*indices)(self.list[0]), itemgetter(*indices)(self.list[1]), itemgetter(*indices)(self.list[2]), itemgetter(*indices)(self.list[3]))

class model:
    def __init__(self, name, seed=0, softmax=False):
        self.name = name
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope(name):
                tf.set_random_seed(seed)
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

                self.input_phb = tf.placeholder(dtype=tf.float32, shape=[None, 2])
                W0b = tf.get_variable(name='W0b', shape=[2, 32], initializer=tf.contrib.layers.xavier_initializer())
                W1b = tf.get_variable(name='W1b', shape=[32, 16], initializer=tf.contrib.layers.xavier_initializer())
                b0b = tf.get_variable(name='b0b', shape=[32], initializer=tf.constant_initializer(0.))
                weights = [W0b, W1b]
                biases = [b0b, b1d]
                outputb = connect(self.input_phb, weights, biases, activations)

                self.input_phc = tf.placeholder(dtype=tf.float32, shape=[None, 2])
                W0c = tf.get_variable(name='W0c', shape=[2, 32], initializer=tf.contrib.layers.xavier_initializer())
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
        print("     %s's mse: %0.3f"%(self.name, mse_run))
        if save:
            self.saver.save(self.sess, './saves/%s_model.ckpt'%(self.name))

    def restore(self):
        self.saver.restore(self.sess, './saves/%s_model.ckpt'%(self.name))

    def calculate_strategy(self, I, A):
        sigma = np.zeros(3)
        d = self.predict(I)[0, A]
        d_plus = np.clip(d, 0, None)
        if d_plus.sum() > 0:
            sigma[A] = d_plus/np.sum(d_plus)
            if np.sum(sigma)!= 1:
                return sigma/np.sum(sigma)
            return sigma
        else:
            sigma[A[np.argmax(d)]] = 1
            return sigma

    def get_strategy(self, I, A):
        sigma = np.zeros(3)
        sigma[A] = self.predict(I)[0, A]
        return sigma

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
        return child

    def set_check(self, child):
        self.check = child
        return child

    def set_bet(self, child):
        self.bet = child
        return child

    def deal(self):
        return self.neighbors[np.random.choice(6)]

    def take(self, action):
        return self.neighbors[action]

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
# 'I' consists of info-set, history, and progress bar.
# 'f' consists of info-set, action0, and action1.
    def __init__(self):
        self.GP_dict = {'0': [1,0,0], '1': [1,0,0], }
        self.perms = ["01", "02", "10", "12", "20", "21"]
        self.n_perms = list(permutations(range(3), 2))
        self.terminal = "EIJGH"
        self.create_history()
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
                    node.I = ([n_perm[node.P]],[0,0],[0,0])
                elif g_node == "C":
                    node.neighbors = [None, node.set_check(tree["E"+perm]), node.set_bet(tree["F"+perm])]
                    node.P = 1
                    node.A = [1,2]
                    node.I = ([n_perm[node.P]],[1,0],[1,0])
                elif g_node == "D":
                    node.neighbors = [node.set_fold(tree["G"+perm]), None, node.set_bet(tree["H"+perm])]
                    node.P = 1
                    node.A = [0,2]
                    node.I = ([n_perm[node.P]],[2,0],[1,0])
                elif g_node == "F":
                    node.neighbors = [node.set_fold(tree["I"+perm]), None, node.set_bet(tree["J"+perm])]
                    node.P = 0
                    node.A = [0,2]
                    node.I = ([n_perm[node.P]],[1,2],[1,1])
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

    def collect_samples(self, node, p, M_r, B_vp, B_s, GP_p):
        if node.name[0] in self.terminal:
            return node.U(p)
        elif node.P == p:
            I = node.I
            A = node.A
            sigma = M_r[p].calculate_strategy(I, A)
            v_a = np.zeros(3)
            for a in A:
                v_a[a] = self.collect_samples(node.take(a), p, M_r, B_vp, B_s, GP_p)
            v_s = np.dot(v_a, sigma)
            d = v_a - v_s
            B_vp.add(I, d)
            GP_p.append(np.append(node.name, d))
            return v_s
        elif node.P == other(p):
            I = node.I
            A = node.A
            sigma = M_r[other(p)].calculate_strategy(I, A)
            B_s.add(I, sigma)
            a = np.random.choice(3, p=sigma)
            return self.collect_samples(node.take(a), p, M_r, B_vp, B_s, GP_p)
        else:
            return self.collect_samples(node.deal(), p, M_r, B_vp, B_s, GP_p)

    def forward_update(self, model, name):
        queue = Queue()
        queue.put(self.root)
        while not queue.empty():
            node = queue.get()
            A = node.A
            p = node.P
            if node.name == "A":
                sigma = np.full(6, 1/6)
            else:
                I = node.I
                sigma = model.get_strategy(I, A)
                self.hist[node.name].append(sigma[A[0]])
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
        with open('sigmas/%s.pkl'%name, 'wb') as output:
            pickle.dump(self.hist, output, pickle.HIGHEST_PROTOCOL)

    def backward_update(self):
        for g_node in "FCDB":
            for perm in self.perms:
                key = g_node + perm
                node = self.tree[key]
                p = node.P
                p_not = other(p)
                # p_not exploits p
                expected = 0
                for a in node.A:
                    neighbor = node.neighbors[a]
                    expected += neighbor.prob[p]*neighbor.value[p_not]
                node.value[p_not] = expected
                # p exploits p_not
                v_a = np.zeros((2,2))
                n_set = []
                g_prob = 0
                a_indices = node.A
                for i in range(2):
                    i_node = self.tree[g_node + self.i_perm(perm, p)[i]]
                    n_set.append(i_node)
                    g_prob += i_node.prob[p_not]
                    for j in range(2):
                        v_a[i,j] = i_node.neighbors[a_indices[j]].value[p]*i_node.prob[p_not]
                v_a = np.sum(v_a, axis = 0)
                n_set[0].value[p] = np.max(v_a)/g_prob
                n_set[1].value[p] = n_set[0].value[p]
        expected = [0,0]
        for neighbor in self.root.neighbors:
            expected[0] += 1/6*neighbor.value[0]
            expected[1] += 1/6*neighbor.value[1]
        self.root.value = expected
        return self.root.value

    def create_history(self):
        self.hist = {}
        for g_node in "BCDF":
            for perm in self.perms:
                self.hist[g_node+perm] = []

    def visualize(self, name):
        hist = pickle.load(open('./sigmas/%s.pkl'%name, 'rb'))
        gn = "BCDF"
        for f in range(4):
            plt.figure(f)
            for s in range(6):
                nn = gn[f]+self.perms[s]
                strat = hist[nn]
                plt.subplot(611+s)
                plt.title(nn)
                plt.ylim(-0.1,1.1)
                plt.plot(strat)
            plt.savefig("./sigmas/%s-%s.png"%(name,gn[f]))
            plt.show()

    def test(self, name):
        M_s = model(name, softmax = True)
        M_s.restore()
        queue = Queue()
        queue.put(self.root)
        while not queue.empty():
            node = queue.get()
            A = node.A
            p = node.P
            if node.name == "A":
                sigma = np.full(6, 1/6)
            else:
                I = node.I
                sigma = M_s.get_strategy(I, A)
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
        for g_node in "FCDB":
            for perm in self.perms:
                key = g_node + perm
                node = self.tree[key]
                p = node.P
                p_not = other(p)
                # p_not exploits p
                expected = 0
                for a in node.A:
                    neighbor = node.neighbors[a]
                    expected += neighbor.prob[p]*neighbor.value[p_not]
                node.value[p_not] = expected
                # p exploits p_not
                v_a = np.zeros((2,2))
                n_set = []
                g_prob = 0
                a_indices = node.A
                for i in range(2):
                    i_node = self.tree[g_node + self.i_perm(perm, p)[i]]
                    n_set.append(i_node)
                    g_prob += i_node.prob[p_not]
                    for j in range(2):
                        v_a[i,j] = i_node.neighbors[a_indices[j]].value[p]*i_node.prob[p_not]
                v_a = np.sum(v_a, axis = 0)
                n_set[0].value[p] = np.max(v_a)/g_prob
                n_set[1].value[p] = n_set[0].value[p]
        expected = [0,0]
        for neighbor in self.root.neighbors:
            expected[0] += 1/6*neighbor.value[0]
            expected[1] += 1/6*neighbor.value[1]
        self.root.value = expected
        print("exploitability\n", self.root.value)
        print("expected exploitability\n", [-1/18,1/18])

def save_W(W):
    with open('saves/W.pkl', 'wb') as output:
        pickle.dump(W, output, pickle.HIGHEST_PROTOCOL)

def save_GP(GP):
    GP0 = pd.DataFrame(GP[0], columns=['name','va0','va1','va2'])
    GP1 = pd.DataFrame(GP[1], columns=['name','va0','va1','va2'])
    print("GP0")
    print(GP0.tail(2))
    print("GP1")
    print(GP1.tail(2))
    GP0.to_csv(r'/scratch/lguo1/DR-2019/saves/GP0.csv')
    GP1.to_csv(r'/scratch/lguo1/DR-2019/saves/GP1.csv')

def connect(input, weights, biases, activations):
    layer = input
    for W, b, activation in zip(weights, biases, activations):
        layer = tf.matmul(layer, W) + b
        if activation is not None:
            layer = activation(layer)
    return layer

def other(p):
    return 1 - p
