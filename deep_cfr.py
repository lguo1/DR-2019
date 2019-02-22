from objects import *
import argparse

def main(iter, trav, train_v=2000, batch_v=1000, train_s=2000, batch_s=1000):
    tf.set_random_seed(1)
    np.random.seed(1)
    G = game()
    B_v = (buffer(), buffer())
    B_s = buffer()
    W = [[],[],[]]
    M_r = (model('p0', True), model('p1', True))
    M_s = model('state')
    for t in range(iter):
        p = t%2
        p_not = (p+1)%2
        B_vp = B_v[p]
        B_s.set()
        B_vp.set()
        for n in range(trav):
            collect_samples(G, "A", p, p_not, M_r, B_vp, B_s)
        W[p].extend([(1+t)/2]*B_vp.count)
        W[2].extend([(1+t)/2]*B_s.count)
        M_r[p].train(B_vp, W[p], train_v, batch_v)
    M_s.train(B_s, W[2], train_s, batch_s, True)

def collect_samples(game, node, p, p_not, M_r, B_vp, B_s):
    if game.is_terminal(node):
        return game.util(node, p)
    elif game.P(node) == p:
        I = game.I(node, p)
        A = game.A(node)
        sigma = calculate_strategy(I, A, M_r[p])
        v_a = np.zeros(3)
        for a in A:
            v_a[a] = collect_samples(game, game.take(node, a), p, p_not, M_r, B_vp, B_s)
        v_s = np.dot(v_a, sigma)
        d = v_a - v_s
        B_vp.add(I, d)
        return v_s
    elif game.P(node) == p_not:
        I = game.I(node, p_not)
        A = game.A(node)
        sigma = calculate_strategy(I, A, M_r[p_not])
        B_s.add(I, sigma)
        try:
            a = np.random.choice(3, p=sigma)
        except ValueError:
            a = np.random.choice(3, p=sigma/sigma.sum())
        return collect_samples(game, game.take(node, a), p, p_not, M_r, B_vp, B_s)
    else:
        return collect_samples(game.deal(), "B", p, p_not, M_r, B_vp, B_s)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("iter", help="number of iterations", type=int, default=10000)
    parser.add_argument("trav", help="number of travesals", type=int, default=100)
    args = parser.parse_args()
    main(int(args.iter), int(args.trav))
