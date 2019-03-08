from objects import *
from itertools import permutations
import argparse

def main(iter, trav, train_v=2000, batch_v=1000, train_s=2000, batch_s=1000):
    tf.set_random_seed(1)
    np.random.seed(1)
    errs = []
    errs = []
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
            G.collect_samples("A", p, p_not, M_r, B_vp, B_s)
        W[p].extend([(1+t)/2]*B_vp.count)
        W[2].extend([(1+t)/2]*B_s.count)
        print("iteration %04d"%t)
        M_r[p].train(B_vp, W[p], train_v, batch_v)
        if t % 100 == 0:
            M_s.train(B_s, W[2], train_s, batch_s, True)
            err0, err1 = measure_performance(M_s)
            errs0.append(err0)
            errs1.append(err1)

def value_state(game, node, p, M_r):
    if game.is_terminal(node):
        return game.util(node, p)
    else:
        sigma = calculate_strategy(I, A, M_r[game.P(node)])
        v_a = np.zeros(3)
        for a in game.A(node):
            v_a[a] = value_state(game, game.take(node, a), p, M_r)
        return np.dot(v_a, sigma)

def best_response(game, node, p, p_not, M_r):
    sigma = np.zeros(3)
    v_a = np.zeros(3)
    A = game.A(node)
    for a in A:
        v_a[a] = value_state(game, game.take(node, a), p, M_r)
    sigma[A[np.argmax(d)]] = 1
    return sigma




def term_util(game, node):
    all_util = np.zeros(6,2)

    all_values = {}
    for i in range(6):
        game.cards = game.all_cards[i]
        all_util[i,0] = value_state(game, node, 0, M_r)
        all_util[i,1] = value_state(game, node, 1, M_r)
        all_values[]

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

def measure_performance(M):
    T = game()
    err0 = 0
    err1 = 0
    sigmas = np.empty([3, 4, 3])
    alphas = np.empty(3)
    for c in range(3):
        for n in range(4):
            node[0] = "BFCD"[n]
            sigmas[c,n] = calculate_strategy(([c], *T.info[node]), T.A(node), M)
    c = 0
    alphas[0] = .5*(sigmas[c, 0, 2] - sigmas[c, 0, 1] + 1)
    err0 += np.sum(np.square(sigmas[c, 1] - np.array([1, 0, 0])))
    err1 += np.sum(np.square(sigmas[c, 2] - np.array([0, 2/3, 1/3])))
    err1 += np.sum(np.square(sigmas[c, 3] - np.array([1, 0, 0])))

    c = 1
    err0 += np.sum(np.square(sigmas[c, 0] - np.array([0, 1, 0])))
    alphas[1] = .5*(sigmas[c, 0, 2] - sigmas[c, 0, 1] + 1/3)
    err1 += np.sum(np.square(sigmas[c, 2] - np.array([0, 1, 0])))
    err1 += np.sum(np.square(sigmas[c, 3] - np.array([2/3, 0, 1/3])))

    c = 2
    alphas[2] = 1/6*(sigmas[c, 0, 2] - sigmas[c, 0, 1] + 1)
    err0 += np.sum(np.square(sigmas[c, 1] - np.array([0, 0, 1])))
    err1 += np.sum(np.square(sigmas[c, 2] - np.array([0, 0, 1])))
    err1 += np.sum(np.square(sigmas[c, 3] - np.array([0, 0, 1])))

    err0 += np.var(alphas)
    return err0, err1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("iter", help="number of iterations", type=int, default=10000)
    parser.add_argument("trav", help="number of travesals", type=int, default=100)
    args = parser.parse_args()
    main(int(args.iter), int(args.trav))
