import tensorflow as tf
import numpy as np
import objects

def main():
    B_r = [[],[]]
    B_s = []
    sess = tf.Session()
    M_r = (model(tf.Graph(), sess, 'p0'), model(tf.Graph(), sess, 'p1'))
    for t in range(iter):
        p = t%2
        p_not = (p+1)%2
        for n in range(trav):
            collect_samples(node(), p, p_not M_r, B_r, B_s, t)
        train_network(B_r[p],0)
        # update time in p_not
    train_network(B_s,1)
    sess.close()
    return

def collect_samples(h, p, p_not M_r B_r, B_s, t):
    if h.is_terminal():
        return h.util(p)
    elif h.P() == p:
        I = h.I(p)
        sigma = calculate_strategy(I, M_r[p])
        v_a = np.zeros(3)
        for a in h.A():
            v_a[a] = collect_samples(h.place(a), p, M_r, B_r, B_s)
        v_s = np.dot(v_a, sigma)
        d = v_a - v_s
        B_r[p].append((I, d, t)) # change
    elif h.P() == p_not:
        sigma = calculate_strategy(I, M_r[p_not])
        B_s.append((h.I(p_not), sigma, t))
        a = np.random.choice(h.A(), p=sigma)
        return collect_samples(h.place(a), p, M_r, B_r, B_s)
    else:
        return collect_samples(h.deal(), p, M_r, B_r, B_s)

def calculate_strategy(I, model):
    sum = 0
    d = model.predict(I)
    d_plus = np.clip(d, 0)
    sum = d_plus.sum()
    if sum > 0:
        return d_plus/sum
    else:
        sigma = np.zeros(3)
        sigma[np.argmax(d)] = 1
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
