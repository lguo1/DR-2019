from games import *
from scipy.linalg import null_space as null

def diff(arr):
    # this function takes an array arr where arr[j] = u^i(ej,a^{-i})
    # it returns a matrix D such that D[k,j] = u^i(ek,a^{-i}) - u^i(ej,a^{-i})
    l = arr.size
    D = np.zeros((l,l))
    for j in range(l-1):
        for k in range(1,l):
            D[j,k]=arr[k]-arr[j]
    return D - np.transpose(D)

def g(ui):
    # this function takes a matrix ui that represents the utility function of player i
    # ui is so set up that the row player is player -i and the column player player i
    # the function returns a 3darray G where G[k,j,y] = u^i(ek,ey) - u^i(ej,ey)
    G = []
    for y in range(ui.shape[0]):
        G.append(diff(ui[y]))
    return np.swapaxes(np.array(G),0,2)

def marginal(Z2):
    # this function takes a joint distribution matrix and returns the product of its marginals.
    z1 = np.sum(Z2, axis=1)[None]
    z2 = np.sum(Z2, axis=0)[None]
    return np.matmul(z1.T,z2)

def prob(nullspace):
    # this function takes a n x m 2darray representing the nullspace of some linear equations.
    # each column in the array is a basis vector.
    # the function returns an n x 1 array which is a probability distribution and belongs to the nullspace.
    nullspace[:,np.all(nullspace<=0,axis=0)] *= -1
    nullspace = nullspace[:,np.logical_not(np.any(nullspace<0,axis=0))]
    p = np.sum(nullspace,axis=1)
    return p/np.sum(p)

def CRM(game, T):
    # this function runs a Universal Calibrated Regret Matching procedure as described in the paper.
    G1 = g(game.u[0])
    G2 = g(game.u[1])
    Z2 = np.zeros(game.shape)
    A1 = game.shape[0]
    A2 = game.shape[1]
    Z2[np.random.choice(A1),np.random.choice(A2)] += 1
    for t in range(1,T+1):
        L1 = np.sum(Z2*G1,axis=2)/t
        L2 = np.sum(np.transpose(Z2)*G2,axis=2)/t
        # the expected result is Li[k,j] = sum(prob(ej,a^{-i})[u^i(ek,a^{-i}) - u^i(ej,a^{-i}])
        # over all a^{-i}s.
        L1[L1<0]=0
        L2[L2<0]=0
        # Li is the lambda matrix described in the paper. It basically is
        # the nonpositive portion of the correlated regret matrix.
        D1 = np.diagflat(np.sum(L1,axis=0))
        D2 = np.diagflat(np.sum(L2,axis=0))
        # Di is a diagonal matrix whose diagonal entries are the sums of Li columns
        q1 = prob(np.around(null(L1-D1),3))
        q2 = prob(np.around(null(L2-D2),3))
        # qi is the strategy played at t+1 that satisfies the condition described
        # in the paper.
        Z2[np.random.choice(A1,p=q1),np.random.choice(A2,p=q2)] += 1
        # Z2 += q1.reshape(2,1)*q2
    print("nonpositive portion of player 1's Correlated Equilibrium matrix\n", L1)
    print("nonpositive portion of player 2's Correlated Equilibrium matrix\n", L2)
    return Z2/T

def check_nash(Gx, x, y):
    # this function is used to verify whether a given strategy is a Nash equilibrium.
    # it takes a matrix G and two strategies for two players.
    # it returns a Nash matrix playing those two strategies.
    return np.sum(np.matmul(x[None].T,y[None])*Gx, axis=2)

def check_corr(Gx, z):
    # this function is used to verify whether a given strategy is a correlated equilibrium.
    # it takes a matrix G and a joint distribution.
    # it returns a correlated matrix playing the joint distribution.
    return np.sum(z*Gx, axis=2)

def NM(game, T):
    # this is the procedure I proposed to find Nash Equilibrium.
    G1 = g(game.u[0])
    G2 = g(game.u[1])
    Z2 = np.zeros(game.shape)
    A1 = game.shape[0]
    A2 = game.shape[1]
    Z2[np.random.choice(A1),np.random.choice(A2)] += 1
    for t in range(1,T+1):
        MZ2 = marginal(Z2/t)
        L1 = np.sum(MZ2*G1,axis=2)
        L2 = np.sum(np.transpose(MZ2)*G2,axis=2)
        L1[L1<0]=0
        L2[L2<0]=0
        # The calculation is similar to CRM except that here we use the product
        # of the marginal distributions rather than the joint distribution.
        # The resultant Li is the Nash matrix where
        # Li[k,j] = z^i(j)z^{-i}(a^{-i})*[u^i(k,a^{-i}) - u^i(j,a^{-i})]
        D1 = np.diagflat(np.sum(L1,axis=0))
        D2 = np.diagflat(np.sum(L2,axis=0))
        q1 = prob(np.around(null(L1-D1),3))
        q2 = prob(np.around(null(L2-D2),3))
        Z2[np.random.choice(A1,p=q1),np.random.choice(A2,p=q2)] += 1
        #Z2 += q1.reshape(2,1)*q2
    print("strategy for player 1: \n",np.sum(Z2, axis=1)[None]/t)
    print("strategy for player 2: \n",np.sum(Z2, axis=0)[None]/t)
    print("nonpositive portion of player 1's Nash matrix\n", L1)
    print("nonpositive portion of player 2's Nash matrix\n", L2)
    print()
    return marginal(Z2/T)

def main():
    np.random.seed(0)
    game = RPS3()
    #print("Nash matrix for the equilbrium strategy \n",check_nash(g(game.u[0]), np.full(3,1/3),np.full(3,1/3)))
    print("joint distribution (row player = player 1) \n", NM(game,2000))

main()
