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
    M_r = (model('p0'), model('p1'))
    M_s = model('state', True)
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
        if (t+1) % iter_per_check == 0:
            M_s.train(B_s, W[2], train_s, batch_s, True)
            G.forward_update(M_s, t)
            print("     exploitability", G.backward_update())
            node = G.tree["D01"]
            I = node.I
            A = node.A
            print(node.name)
            print(I)
            print("     l_d", M_s.predict(I)[0])
            print("     sigma", M_s.calculate_strategy(I, A))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("iter", help="number of iterations", type=int, default=10000)
    parser.add_argument("trav", help="number of travesals", type=int, default=100)
    parser.add_argument("--train_s", type=int, default=2000)
    parser.add_argument("--batch_s", type=int, default=1000)
    parser.add_argument("--train_v", type=int, default=2000)
    parser.add_argument("--batch_v", type=int, default=1000)
    parser.add_argument("--check_freq", type=int, default=100)
    args = parser.parse_args()
    main(int(args.iter), int(args.trav), train_s = int(args.train_s), train_v = int(args.train_v)
    , batch_v = int(args.train_s), batch_s = int(args.train_s), iter_per_check = int(args.check_freq))
