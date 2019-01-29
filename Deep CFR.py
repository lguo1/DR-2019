import tensorflow as tf
import numpy as np

def main():
    B_p = []
    B_s = []
    models = (model(tf.Graph(), 'p1'), model(tf.Graph(), 'p2'))
    for t in range(iter):
        p = t%2
        p_not = (p+1)%2
        for n in range(trav):
            collect_samples([], p, models, B_p, B_s)
        train_network(B_p[p],0)
        B_p[p_not] =
    train_network(B_s,1)
    return

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

def collect_samples(h, p, models B_p, B_s):
    if h.is_terminal():
        return util(h, p)
    elif P(h) == p:
        sigma = calculate_strategy(I(h), models[p])
        v = [0]
        for a in A(h):
            v(a) = collect_samples(h*a, p, models, B_p, B_s)
            v += sigma[a]+v(a)
        for a in A(h):
            d[I][a] = v(a) - v
        Add(B_p, (I, d[I], t))
    elif P(h) = p_not:
        a = np.random.choice(A(h), sigma)
        return collect_samples(h*a, p, models, B_p, B_s)
    else:
        a = np.random.choice(A(h), sigma)
        return collect_samples(h*a, p, models, B_p, B_s)

def calculate_strategy(I, m_p):
    sum = 0
    D = f(I,m_p)
    for a in A(I):
        sum += max(0, D(I, a))
    if sum > 0:
        for a in A(I):
            sigma = max(0, D(I, a))/sum
    else:
        for a in A(I):
            sigma = [0]
        sigma[argmax(D(I,a))] = 1
    return sigma

def train_network(B, S):
    for b in range(train):
        for i in range(batch):
            sample(B)
            regret = f(I, model)
            if S:
                for a in A:
                    y[a] = np.exp(regret[a])/sum(regret[a])
            else:
                y = regret
        L = mse(batch)
        adam_optimizer()
    return model
