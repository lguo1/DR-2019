from deep_cfr import *

M = model("state")
M.restore()
T = game()
T.deal()
def mse_p0():
    T.cards = np.array([[1],[2]])
    strat = M.predict(T.I("B", 0))
    op_strat = np.array([0, 1, 0])
    print(strat)
    print(op_strat)
    return np.sum(np.square(strat - op_strat))

def check_strategy(T, M):
    p0 = []
    p1 = []
    T.cards = np.array([[1],[0]])
    print("cards:", T.cards[0,0], T.cards[1,0])
    for node in ["B", "F"]:
        p0.append(calculate_strategy(T.I(node, 0), T.A(node), M))
    for node in ["C", "D"]:
        p1.append(calculate_strategy(T.I(node, 1), T.A(node), M))
    print("p0 strats: \nB:", p0[0], "\nF:", p0[1])
    print("p1 strats: \nC:", p1[0], "\nD:", p1[1])

def check_collect_samples(T, M):
    T.cards = np.array([[1],[0]])
    print("cards:", T.cards[0,0], T.cards[1,0])
    B_v = (buffer(), buffer())
    B_s = buffer()
    d_p0 = {}
    d_p1 = {}
    for node in list(T.tree.keys()):
        d_p0[node] = collect_samples(T, node, 0, 1, (M, M), B_v[0], B_s)
        d_p1[node] = collect_samples(T, node, 1, 0, (M, M), B_v[1], B_s)
    print("p = p0 samples:", d_p0)
    print(d_p0["B"])
    sigma = calculate_strategy(T.I("B", 0), T.A("B"), M)
    print(d_p0["D"]*sigma[2] + d_p0["C"]*sigma[1])
    print("p = p1 samples:", d_p1)
    T.cards = np.array([[2],[1]])
    print("cards:", T.cards[0,0], T.cards[1,0])
    B_v = (buffer(), buffer())
    B_s = buffer()
    d_p0 = {}
    d_p1 = {}
    for node in list(T.tree.keys()):
        d_p0[node] = collect_samples(T, node, 0, 1, (M, M), B_v[0], B_s)
        d_p1[node] = collect_samples(T, node, 1, 0, (M, M), B_v[1], B_s)
    print("p = p0 samples:", d_p0)
    print("p = p1 samples:", d_p1)
    # print(B_v[0].list)
    # print(B_v[1].list)
    # print(B_s.list)

check_strategy(T, M)
check_collect_samples(T, M)
#print(mse_p0())
