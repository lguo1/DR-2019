from objects import *
from itertools import permutations
import argparse

def main(iter, trav, train_v=2000, batch_v=1000, train_s=2000, batch_s=1000, iter_per_check=100):
    tf.set_random_seed(1)
    np.random.seed(1)
    G = Game()
    B_v = (buffer(), buffer())
    B_s = buffer()
    W = [[],[],[]]
    M_r = (model('p0', True), model('p1', True))
    M_s = model('state')
    for t in range(iter):
        p = t%2
        B_vp = B_v[p]
        B_s.set()
        B_vp.set()
        for n in range(trav):
            G.collect_samples(G.root, p, M_r, B_vp, B_s)
        W[p].extend([(1+t)/2]*B_vp.count)
        W[2].extend([(1+t)/2]*B_s.count)
        '''
        if p == 1:
            node = G.tree["D01"]
            I = node.I(p)
            A = node.A
            print(">>>>>>>>")
            print(node.name)
            print("l_d", M_r[p].predict(I)[0])
            print("sigma", M_r[p].calculate_strategy(I, A))
            print("<<<<<<<<")
        else:
            node = G.tree["F20"]
            I = node.I(p)
            A = node.A
            print(node.name)
            print("l_d", M_r[p].predict(I)[0])
            print("sigma", M_r[p].calculate_strategy(I, A))
            print("_____")
        '''
        if (t+1) % iter_per_check == 0:
            M_s.train(B_s, W[2], train_s, batch_s, True)
            G.forward_update(M_s, t)
            print("     exploitability:", G.backward_update())

'''
def measure_performance(M):
    T = Game()
    err0 = 0
    err1 = 0
    sigmas = np.empty([3, 4, 3])
    alphas = np.empty(3)
    for c in range(3):
        for n in range(4):
            node[0] = "BFCD"[n]
            sigmas[c,n] = M.calculate_strategy(([c], *T.info[node]), T.A(node))
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
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("iter", help="number of iterations", type=int, default=10000)
    parser.add_argument("trav", help="number of travesals", type=int, default=100)
    parser.add_argument("--train_s", type=int, default=2000)
    parser.add_argument("--batch_s", type=int, default=1000)
    parser.add_argument("--train_v", type=int, default=2000)
    parser.add_argument("--batch_v", type=int, default=1000)
    parser.add_argument("--iter_per_check", type=int, default=100)
    args = parser.parse_args()
    main(int(args.iter), int(args.trav), train_s = int(args.train_s), train_v = int(args.train_v)
    , batch_v = int(args.train_s), batch_s = int(args.train_s), iter_per_check = int(args.iter_per_check))
