from objects import *

T = game()
T.cards = np.array([[1],[2]])
M = model("state")
M.restore()
strat = M.predict(T.I("B", 0))
op_strat = np.array([0, 1, 0])
print(np.sum(np.square(strat - op_strat)))
