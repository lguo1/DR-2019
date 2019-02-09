class model:
    def __init__(self, graph, sess, name, softmax=True):
        with graph.as_default():
            with tf.variable_scope(name)):
                self.input_ph_0 = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                self.input_ph_1 = tf.placeholder(dtype=tf.float32, shape=[None, 3, 2])
                self.output_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3])

                W0_0 = tf.get_variable(name='W0_0', shape=[1, 64], initializer=tf.contrib.layers.xavier_initializer())
                W1_0 = tf.get_variable(name='W1_0', shape=[64, 64], initializer=tf.contrib.layers.xavier_initializer())

                W0_1 = tf.get_variable(name='W0_1', shape=[3,2,], initializer=tf.contrib.layers.xavier_initializer())
                W1_1 = tf.get_variable(name='W1_1', shape=[20, 1], initializer=tf.contrib.layers.xavier_initializer())

                b0 = tf.get_variable(name='b0', shape=[20], initializer=tf.constant_initializer(0.))
                b1 = tf.get_variable(name='b1', shape=[20], initializer=tf.constant_initializer(0.))
                b2 = tf.get_variable(name='b2', shape=[1], initializer=tf.constant_initializer(0.))

                self.weights = [W0, W1, W2]
                self.biases = [b0, b1, b2]
                if softmax:
                    self.activations = [tf.nn.relu, tf.nn.relu, tf.nn.softmax]
                else:
                    self.activations = [tf.nn.relu, tf.nn.relu, None]

                layer = self.input_ph
                for W, b, activation in zip(self.weights, self.biases, self.activations):
                    layer = tf.matmul(layer, W) + b
                    if activation is not None:
                        layer = activation(layer)

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
        self.bets = np.zeros(3)
        self.turns = -1
        self.is_terminal = False

    def deal(self):
        self.cards = np.random.choice(3,2)
        self.turns += 1

    def place(self, arr):
        self.bets[self.turns] = arr
        self.turns += 1

    def A(self):
        if self.turns == 0:
            return [[1,1],[2,1]]
        elif self.turns == 1:
            if self.bets[0] == [1,1]:
                return [[1,1],[2,1]]
            else:
                return [[0,1],[2,1]]
        else:
            return [[0,1],[2,1]]

    def P(self):
        if self.turns == -1:
            return -1
        else:
            return (self.turns+1)%2

    def is_terminal(self):
        if self.bets[2] != [0,0]:
            return True
        elif self.bets[1] == [0,0]:
            return False
        elif self.bets[1] != [2,1]:
            return True
        elif self.bets[0] == [2,1]:
            return True
        else:
            return False

    def util(self, p):
        thd = self.bets[2]
        if thd == [0,1]:
            return [0,3][p]
        elif thd == [2,1]:
            return (p == np.argmax(self.cards))*4
        else:
            sec = self.bets[1]
            if sec == [1,1]:
                return (p == np.argmax(self.cards))*2
            elif sec == [2,1]:
                return (p == np.argmax(self.cards))*4
            else:
                return [3,0][p]

    def I(self, p):
        return (self.cards[p], self.bets)
