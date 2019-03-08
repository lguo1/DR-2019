from objects import *
from itertools import permutations
import argparse

def main(iter, trav, train_v=2000, batch_v=1000, train_s=2000, batch_s=1000):
    tf.set_random_seed(1)
    np.random.seed(1)
    errs0 = []
    errs1 = []
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
        print("iteration %04d"%t)
        M_r[p].train(B_vp, W[p], train_v, batch_v)
        if t % 100 == 0:
            M_s.train(B_s, W[2], train_s, batch_s, True)
            err0, err1 = measure_performance(M_s)
            errs0.append(err0)
            errs1.append(err1)

def value_state(game, node, p, M_r):
    if node in game.terminal:
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

def exp_tree(game, node, p, p_not, M_r, cards_i, strat, exp_tree):
    util_tree = {}
    F_util = np.zeros((6, 2))
    sigma = np.zeros((6, 2))
    for i in range(6):
        perm = game.all_perms[i]
        game.cards = np.array([perm])
        util = (game.util("I", 0), game.util("J", 0))
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

def strat_tree(game, M_r):
    strat_tree = {}
    strat_per_node = np.zeros(3,3)
    for node in "BCDF":
        for hand in range(3):
            p = game.P(node)
            game.cards[p] = hand
            strat_per_node[hand] = calculate_strategy(game.I(node, p), game.A(node), M_r[p])
        strat[node] = strat_per_node
    return strat


def term_util(game, node):
    all_util = np.zeros(6,2)

    all_values = {}
    for i in range(6):
        game.cards = game.all_cards[i]
        all_util[i,0] = value_state(game, node, 0, M_r)
        all_util[i,1] = value_state(game, node, 1, M_r)
        all_values[]


def collect_samples(game, node, p, p_not, M_r, B_vp, B_s):
    if node in game.terminal:
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

def measure_performance(M):
    T = game()
    err0 = 0
    err1 = 0
    sigmas = np.empty([3, 4, 3])
    alphas = np.empty(3)
    for c in range(3):
        for n in range(4):
            node = "BFCD"[n]
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
