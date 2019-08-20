from games import *
from scipy.linalg import null_space as null

def update_w(W,U,x,y):
    W[:,x] += U[y]-U[y,x]

def CRM(game, T):
    Z2 = np.zeros(game.shape)
    U1 = game.u[0]
    U2 = game.u[1]
    A1 = game.shape[0]
    A2 = game.shape[1]
    W1 = np.zeros((A1,A1))
    W2 = np.zeros((A2,A2))
    a1 = np.random.choice(A1)
    a2 = np.random.choice(A2)
    Z2[a1,a2] += 1
    update_w(W1,U1,a1,a2)
    update_w(W2,U2,a2,a1)
    for t in range(1,T+1):
        L1 = np.clip(W1,0,None)/t
        L2 = np.clip(W2,0,None)/t
        D1 = np.diagflat(np.sum(L1,axis=0))
        D2 = np.diagflat(np.sum(L2,axis=0))
        q1 = np.sum(null(L1-D1),axis=1)
        q1 /= np.sum(q1)
        q2 = np.sum(null(L2-D2),axis=1)
        q2 /= np.sum(q2)
        a1 = np.random.choice(A1,p=q1)
        a2 = np.random.choice(A2,p=q2)
        Z2[a1,a2] += 1
        update_w(W1,U1,a1,a2)
        update_w(W2,U2,a2,a1)
    return Z2/T

def main():
    np.random.seed(0)
    print(CRM(Chicken(),100))

main()
