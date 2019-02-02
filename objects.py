class model:
    def __init__(self, graph, sess, name, softmax=True):
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
                if softmax:
                    self.output_pred = tf.exp(layer)/tf.exp(layer).sum()
                else:
                    self.output_pred = layer
                self.sess = sess

    def predict(self, inputs):
        return self.sess.run(self.output_pred, feed_dict={input_ph: inputs})

    def train(self, B, S, N_train, N_batch):
        mse = tf.reduce_mean(0.5 * tf.square(self.output_pred - self.output_ph))
        opt = tf.train.AdamOptimizer().minimize(mse)
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for training_step in range(N_train):
            indices = np.random.randint(low=0, high=len(B), size=N_batch)
            input_batch = B[indices]
            output_batch = outputs[indices] # change
            _, mse_run = sess.run([opt, mse], feed_dict={self.input_ph: input_batch, self.output_ph: output_batch})
            if training_step % 1000 == 0:
                print('{0:04d} mse: {1:.3f}'.format(training_step, mse_run))
                saver.save(sess, '/saved/%s_%d.ckpt'%(self.name, training_step))


class node:
    def __init__(self):
        self.bets = np.full(3,np.inf)
        self.turns = -1
        self.is_terminal = False

    def deal(self):
        self.cards = np.random.choice(3,2)

    def place(self, arr):
        self.bets[self.turns] = arr

    def A(self):
        if self.turns == 0:
            return [1,2]
        elif self.turns == 1:
            if self.bets[0] == 1:
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
        if self.bets[2] != np.inf:
            return True
        elif self.bets[1] != np.inf:
            return True
        else:
            return False

    def util(self, p):
        thir = self.bets[2]
        if third == 0:
            return [3,0][p]
        elif third == 2:
            return (p == np.argmax(self.cards))*4
        else:
            sec = self.bets[1]
            if sec == 1:
                return (p == np.argmax(self.cards))*2
            elif sec == 2:
                return (p == np.argmax(self.cards))*4
            else:
                return (p == np.argmax(self.cards))*3

    def I(self, p):
        return (self.cards[p], self.bets)
