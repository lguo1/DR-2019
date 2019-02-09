from objects import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("iter", help="number of iterations", type=int, default=10000)
parser.add_argument("trav", help="number of iterations", type=int, default=100)
args = parser.parse_args()

def main(iter, trav, N_train=2000, N_batch=1000):
    tf.set_random_seed(1)
    np.random.seed(1)

    B_v = (buffer(), buffer())
    B_s = buffer()
    W_v = [[],[]]
    M_r = (model('p0', True), model('p1', True))
    M_s = model('state')
    for t in range(iter):
        p = t%2
        p_not = (p+1)%2
        B_vp = B_v[p]
        B_vp.set()
        for n in range(trav):
            collect_samples(node(), p, p_not, M_r, B_vp, B_s)
        W_v[p].extend([(1+t)/2]*B_vp.count)
        M_r[p].train(B_vp, W_v[p], N_train, N_batch)
    M_s.train(B_s)
    sess.close()
    return

def collect_samples(h, p, p_not, M_r, B_vp, B_s):
    if h.is_terminal():
        return h.util(p)
    elif h.P() == p:
        I = h.I(p)
        A = h.A()
        sigma = calculate_strategy(I, A, M_r[p])
        v_a = np.zeros(3)
        for a in A:
            v_a[a] = collect_samples(h.place(a), p, p_not, M_r, B_vp, B_s)
        v_s = np.dot(v_a, sigma)
        d = v_a - v_s
        B_vp.add(I,d)
    elif h.P() == p_not:
        I = h.I(p_not)
        A = h.A()
        sigma = calculate_strategy(I, A, M_r[p_not])
        B_s.add(I, sigma)
        try:
            a = np.random.choice(3, p=sigma)
        except ValueError:
            a = np.random.choice(3, p=sigma/sigma.sum())
        return collect_samples(h.place(a), p, p_not, M_r, B_vp, B_s)
    else:
        return collect_samples(h.deal(), p, p_not, M_r, B_vp, B_s)

def calculate_strategy(I, A, model):
    sigma = np.zeros(3)
    d = model.predict(I)[0, A]
    d_plus = np.clip(d, 0, None)
    if d_plus.sum() > 0:
        sigma[A] = d_plus/d_plus.sum()
        return sigma
    else:
        sigma[A[np.argmax(d)]] = 1
        return sigma

main(int(args.iter), int(args.trav))
