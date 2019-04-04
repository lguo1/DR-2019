from objects import *
import argparse

def main(iter, trav, seed=1, train_v=2000, batch_v=1000, train_s=2000, batch_s=1000, check_freq=100):
    np.random.seed(seed)
    G = Game()
    B_v = (buffer(), buffer())
    B_s = buffer()
    W = [[],[],[]]
    M_r = (model('p0', seed), model('p1', seed))
    M_s = model('state', seed, True)
    for t in range(iter):
        p = t%2
        B_vp = B_v[p]
        B_s.set()
        B_vp.set()
        for n in range(trav):
            G.collect_samples(G.root, p, M_r, B_vp, B_s)
        W[p].extend([(1+t)/2]*B_vp.count)
        W[2].extend([(1+t)/2]*B_s.count)
        print("iteration %04d"%t)
        M_r[p].train(B_vp, W[p], train_v, batch_v)
        if (t+1) % check_freq == 0:
            M_s.train(B_s, W[2], train_s, batch_s, True)
            G.forward_update(M_s, t)
            print("     exploitability", G.backward_update())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("iter", help="number of iterations", type=int, default=10000)
    parser.add_argument("trav", help="number of travesals", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train_s", type=int, default=2000)
    parser.add_argument("--batch_s", type=int, default=1000)
    parser.add_argument("--train_v", type=int, default=2000)
    parser.add_argument("--batch_v", type=int, default=1000)
    parser.add_argument("--check_freq", type=int, default=100)
    args = parser.parse_args()
    main(int(args.iter), int(args.trav), seed = int(args.seed), train_s = int(args.train_s), train_v = int(args.train_v)
    , batch_v = int(args.train_s), batch_s = int(args.train_s), iter_per_check = int(args.check_freq))
