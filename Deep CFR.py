import tensorflow as tf
import numpy as np
import objects

def main():
    B_p = []
    B_s = []
    models = (model(tf.Graph(), 'p0'), model(tf.Graph(), 'p1'))
    for t in range(iter):
        p = t%2
        p_not = (p+1)%2
        for n in range(trav):
            collect_samples(node(), p, models, B_p, B_s)
        train_network(B_p[p],0)
        B_p[p_not] =
    train_network(B_s,1)
    return

def collect_samples(h, p, models B_p, B_s):
    if h.is_terminal():
        return h.util(p)
    elif h.P() == p:
        sigma = calculate_strategy(h.I(p), models[p])
        v = [0]
        for a in h.A():
            v(a) = collect_samples(h.place(a), p, models, B_p, B_s)
            v += sigma[a]+v(a)
        for a in h.A():
            d[I][a] = v(a) - v
        Add(B_p, (I, d[I], t))
    elif h.P() == p_not:
        a = np.random.choice(h.A(), sigma)
        return collect_samples(h.place(a), p, models, B_p, B_s)
    else:
        return collect_samples(h.deal(), p, models, B_p, B_s)

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
