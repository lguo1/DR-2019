class model:
    def __init__(self, graph, name):
        with graph.as_default():
            with tf.variable_scope(name)):
                self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                self.output_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])

                W0 = tf.get_variable(name='W0', shape=[1, 20], initializer=tf.contrib.layers.xavier_initializer())
                W1 = tf.get_variable(name='W1', shape=[20, 20], initializer=tf.contrib.layers.xavier_initializer())
                W2 = tf.get_variable(name='W2', shape=[20, 1], initializer=tf.contrib.layers.xavier_initializer())

                b0 = tf.get_variable(name='b0', shape=[20], initializer=tf.constant_initializer(0.))
                b1 = tf.get_variable(name='b1', shape=[20], initializer=tf.constant_initializer(0.))
                b2 = tf.get_variable(name='b2', shape=[1], initializer=tf.constant_initializer(0.))

                self.weights = [W0, W1, W2]
                self.biases = [b0, b1, b2]
                self.activations = [tf.nn.relu, tf.nn.relu, None]

                layer = self.input_ph
                for W, b, activation in zip(self.weights, self.biases, self.activations):
                    layer = tf.matmul(layer, W) + b
                    if activation is not None:
                        layer = activation(layer)
                self.output_pred = layer

class node:
    def __init__(self):
        self.bets = np.zeros(3)
        self.turns = -1
        self.is_terminal = False

    def deal(self):
        self.cards = np.random.choice(3,2)

    def place(self, arr):
        self.bets[self.turns]=arr

    def A(self):
        if self.turns == 0:
            return [2,3]
        elif self.turns == 1:
            if self.bets[0] == 2:
                return [2,3]
            else:
                return [1,3]
        else:
            return [1,3]

    def P(self):
        if self.turns == -1:
            return -1
        else:
            return (self.turns+1)%2

    def is_terminal(self):
        if self.bets[2] != 0:
            return True
        elif self.bets[1] != 0:
            return True
        else:
            return False

    def util(self, p):
        thir = self.bets[2]
        if third == 1:
            return [3,0][p]
        elif third == 3:
            return (p == np.argmax(self.cards))*4
        else:
            sec = self.bets[1]
            if sec == 2:
                return (p == np.argmax(self.cards))*2
            elif sec == 3:
                return (p == np.argmax(self.cards))*4
            else:
                return (p == np.argmax(self.cards))*3
