from games import *
from scipy.linalg import null_space as null

def diff(arr):
    l = arr.size
    D = np.zeros((l,l))
    for j in range(l-1):
        for k in range(1,l):
            D[j,k]=arr[k]-arr[j]
    return D - np.transpose(D)

def g(ui):
    G = []
    for y in range(ui.shape[0]):
        G.append(diff(ui[y]))
    return np.swapaxes(np.array(G),0,2)

def CRM(game, T):
    G1 = g(game.u[0])
    G2 = g(game.u[1])
    Z2 = np.zeros(game.shape)
    A1 = game.shape[0]
    A2 = game.shape[1]
    Z2[np.random.choice(A1),np.random.choice(A2)] += 1
    for t in range(1,T+1):
        L1 = np.sum(Z2*G1,axis=2)/t
        L2 = np.sum(np.transpose(Z2)*G2,axis=2)/t

        print(t)
        print("Z2\n",Z2)
        print("G2\n",G2)
        print("Z1*G2\n",np.transpose(Z2)*G2)
        print("L2\n",L2)
        L1[L1<0]=0
        L2[L2<0]=0
        D1 = np.diagflat(np.sum(L1,axis=0))
        D2 = np.diagflat(np.sum(L2,axis=0))
        q1 = np.sum(null(L1-D1),axis=1)
        q1 /= np.sum(q1)
        q2 = np.sum(null(L2-D2),axis=1)
        q2 /= np.sum(q2)

        print("D2\n",D2)
        print("null2\n", null(L2-D2))
        print("q2\n",q2)
        print()

        Z2[np.random.choice(A1,p=q1),np.random.choice(A2,p=q2)] += 1
        #Z2 += q1.reshape(2,1)*q2
    return Z2/T

def main():
    np.random.seed(0)
    game = Chicken()
    print(CRM(game,100))

main()
