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

# fold 0; check 1; bet 2.
class node:
    def __init__(self):

class game:
    def __init__(self):
        self.tree = {}
        for card in self.perms:
            for g_node in "BCDEFGHIJ":
                self.tree["%s%s"%(g_node, card)] = node()
        self.set_up()

        self.perms= ["01", "02", "10", "12", "20", "21"]
        self.tree = {}
        perm_lst = []
        for perm in self.perms:
            perm_lst.append("B" + perm)
            build_subtree(tree, perm)
        self.tree["A": perm_lst]

        self.terminal = "EIJGH"

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

        self.p0_set = [["01", "02"], ["10", "12"], ["20", "21"]]
        self.p1_set = [["10", "20"], ["01", "21"], ["02", "12"]]

        self.available = {
        "B": [1,2],
        "C": [1,2],
        "D": [0,2],
        "F": [0,2]
        }

        self.inheritance = {
        "C": "B1",
        "D": "B2",
        "E": "C1",
        "F": "C2",
        "G": "D0",
        "H": "D2",
        "I": "F0",
        "J": "F2"
        }

    def set_up():
        for key in self.tree:
            g_node = key[0]
            card = key[1:3]
            if g_node == "B":
                self.tree[key].check = self.



    def build_n_tree():
        tree = {
        A = np.zeros(6)
        }
        for perm in self.perms:
            build_n_subtree(tree, perm)
        return n_tree

    def deal(self):
         return self.tree["A"][np.random.choice(6, 1)]

    def util(self, node, p):
        if node[0] == "E":
            return (p == np.argmax(self.cards))*2-1
        elif node[0] == "I":
            return [-1,1][p]
        elif node[0] == "J":
            return (p == np.argmax(self.cards))*4-2
        elif node[0] == "G":
            return [1,-1][p]
        elif node[0] == "H":
            return (p == np.argmax(self.cards))*4-2
        else:
            return None

    def take(self, node, action):
        return self.tree[node][action]

    def A(self, node):
        return self.available[node[0]]

    def P(self, node):
        if node[0] in ["C", "D"]:
            return 1
        elif node[0] in ["B", "F"]:
            return 0
        else:
            return None

    def I(self, node, p):
        return (node[p+1], *self.info[node[0]])

    def is_terminal(self, node):
        return node[0] in self.terminal

    def collect_samples(self, node, p, p_not, M_r, B_vp, B_s):
        if self.is_terminal(node):
            return self.util(node, p)
        elif self.P(node) == p:
            I = self.I(node, p)
            A = self.A(node)
            sigma = calculate_strategy(I, A, M_r[p])
            v_a = np.zeros(3)
            for a in A:
                v_a[a] = self.collect_samples(self.take(node, a), p, p_not, M_r, B_vp, B_s)
            v_s = np.dot(v_a, sigma)
            d = v_a - v_s
            B_vp.add(I, d)
            return v_s
        elif self.P(node) == p_not:
            I = self.I(node, p_not)
            A = self.A(node)
            sigma = calculate_strategy(I, A, M_r[p_not])
            B_s.add(I, sigma)
            try:
                a = np.random.choice(3, p=sigma)
            except ValueError:
                a = np.random.choice(3, p=sigma/sigma.sum())
            return self.collect_samples(self.take(node, a), p, p_not, M_r, B_vp, B_s)
        else:
            return self.collect_samples(self.deal(), p, p_not, M_r, B_vp, B_s)

    def parent(self, child):
        if child[0] == "B":
            return "A", self.perms.index(child[1:3])
        else:
            return self.inheritance[child[0]]

    def build_strat(self, M_r):
        queue = Queue()
        strat_p0 = {"A": 1}
        strat_p1 = {"A": 1}
        queue.enqueue("A")
        while not queue.is_empty():
            node = queue.dequeue()
            p = self.P(node)
            A = self.A(node)
            I = self.I(node, p)
            sigma = calculate_strategy(I, A, M_r[p])
            for a in A:
                neighbor = self.take(node, a)
                if not neighbor.is_terminal():
                    queue.enqueue(neighbor)
                if neighbor[0] == "B":
                    strat_p0[neighbor] = 1/6
                    strat_p1[neighbor] = 1/6
                elif p == 0:
                    strat_p0[neighbor] = strat_p0[node]*sigma[a]
                    strat_p1[neighbor] = strat_p1[node]
                elif p == 1:
                    strat_p0[neighbor] = strat_p1[node]
                    strat_p1[neighbor] = strat_p1[node]*sigma[a]
        return strat_p0, strat_p1

    def exploit_p0(self, node, strat_p0):
        n_tree = self.build_n_tree()
        prob = np.zeros(2)
        sigmas = np.zeros(3,3)
        for level in ["IJ", "EFGH", "CD", "B", "A"]:
            for key in level:
                for card in self.perms:
                    node = key + card
                    if self.is_terminal(node):
                        parent = self.parent(node)
                        n_tree[parent[0]][parent[1]] = self.util(node, 1)
                    elif self.P(node) == 0:
                        parent = self.parent(node)
                        n_tree[parent[0]][parent[1]] = np.dot(n_tree[node], strat_p0[node])
                    elif self.P(node) == 1:
                        probs = np.zeros(2)
                        for i in range(2):
                            i_node = key + card[self.p1_set[node[2][i]]
                            probs[i] = strat_p0[i_node]
                        probs = probs/probs.sum()
                        for i in range(2):
                            parent = self.parent(node)

                        parent = self.parent(node)
                        n_tree[parent[0]][parent[1]] = np.max(n_tree[node])
                    else:
                        raise




        v_tree = {}
        F_util = np.zeros((6, 2))
        sigma = np.zeros((6, 2))
        for i in range(6):
            perm = game.all_perms[i]
            game.cards = np.array([perm])
            value = (game.util("I", 0), game.util("J", 0))
            sigma[i, np.argmax(util)] = strat["C"][perm[1], 2]/
            per_p = per_p_not
        elif game.P(node) == p:
            for a in game.A(node):
                next = game.take(node, a)
                exp_tree[next][p, cards_i] = strat[node][game.all_cards[cards_i][p]][a]*exp_tree[node][p, cards_i]
                exploit(game, next, cards_i, strat, exp_tree)
        elif game.P(node) == p_not:
            for a in game.A(node):
                next = game.take(node, a)
                exp_tree[next][p, cards_i] = strat[node][game.all_cards[cards_i][p]][a]
                exploit(game, next, cards_i, strat, exp_tree)



def connect(input, weights, biases, activations):
    layer = input
    for W, b, activation in zip(weights, biases, activations):
        layer = tf.matmul(layer, W) + b
        if activation is not None:
            layer = activation(layer)
    return layer
def build_subtree(tree, key):
    tree["B" + key] = ["", "C" + key, "D" + key]
    tree["C" + key] = ["", "E" + key, "F" + key]
    tree["D" + key] = ["G" + key, "", "H" + key]
    tree["F" + key] = ["I" + key, "", "J" + key]

def build_n_subtree(tree, key):
    tree["B" + key] = np.zeros(3)
    tree["C" + key] = np.zeros(3)
    tree["D" + key] = np.zeros(3)
    tree["F" + key] = np.zeros(3)
