from exp_objects import *
import time
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel as W, ConstantKernel as C, Matern, RBF, ExpSineSquared as ESS, RationalQuadratic as RQ, DotProduct as DP
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('n_samples',type=int)
args = parser.parse_args()

def translate(raw, pre, post, cols, p):
    translated = []
    for x in raw:
        translated.append(pre[x[0]] + post[x[p+1]])
    return pd.DataFrame(translated, columns=cols, index=raw.index)

def main(n_samples):
    np.random.seed(0)
    #pre = {'B':[0,0,0,0],'C':[1,0,1,0],'D':[2,0,1,0],'F':[1,2,1,1]}
    #post = {'0': [1,0,0],'1': [0,1,0], '2':[0,0,1]}
    #cols = ['pre0','pre1', 'pre2', 'pre3', 'post0', 'post1', 'post2']
    pre = {'B':[0], 'C':[1], 'D':[2], 'F': [3]}
    post = {'0': [0], '1': [1], '2': [2]}
    cols = ['pre0', 'post0']
    print('player 0')
    data0 = pd.read_csv("./saves/GP0.csv", index_col=0)
    print('data\n%s'%(data0.tail(2)))
    raw = data0[-n_samples:]['name']
    translation = translate(raw, pre, post, cols, 0)
    print('translation\n%s'%(translation.tail(2)))
    X0_train, X0_test, y0_train, y0_test = train_test_split(translation, data0[-n_samples:][['va0','va1','va2']])
    # kernel = W() + C(10, (1e-3, 1e1)) + C(1, (1e-3, 1e1)) * RBF(10, (1e-3, 1e1)) + C(1, (1e-3, 1e1)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e1), nu=1.5) + C(1, (1e-3, 1e1)) * RationalQuadratic(length_scale=1.0, alpha=0.1)
    # kernel = W() + C(1, (1e-3, 1e1)) + C(1, (1e-3, 1e1)) * RBF(1, (1e-3, 1e1)) + C(1, (1e-3, 1e1)) * ESS(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + C(1, (1e-3, 1e1)) * DP() + C(1, (1e-3, 1e1)) * ESS(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) * RBF(1, (1e-3, 1e1)) + C(1, (1e-3, 1e1)) * ESS(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) * DP() + C(1, (1e-3, 1e1)) * RBF(1, (1e-3, 1e1)) * DP() + C(1, (1e-3, 1e1)) * ESS(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) * RBF(1, (1e-3, 1e1)) * DP()
    # kernel = W() + C(1, (1e-3, 1e1)) + C(1, (1e-3, 1e1)) * RationalQuadratic(length_scale=1.0, alpha=0.1) + ESS(1.0, 2.0, periodicity_bounds=(1e-2, 1e2)) * DP() + C(1, (1e-3, 1e1)) * ESS(1.0, 2.0, periodicity_bounds=(1e-2, 1e2))
    kernel = W() + C(1, (1e-3, 1e1)) + C(1, (1e-3, 1e1)) * RBF(1, (1e-3, 1e1)) +  ESS(1.0, 2.0, periodicity_bounds=(1e-2, 1e1)) * DP()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    stime = time.time()
    gp.fit(X0_train, y0_train)
    print("\nTime for GPR fitting: %.3f" % (time.time() - stime))
    print("Posterior (kernel: %s)\n Log-Likelihood: %.3f" % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)))
    print('training result: %f'%r2_score(y0_train, gp.predict(X0_train)))
    print('test result: %f'%r2_score(y0_test, gp.predict(X0_test)))

main(args.n_samples)

'''
W = pickle.load(open('./saves/W.pkl', 'rb'))
length = X0_train.shape[0]
W0 = np.array(W[0][-length:])

param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3], "kernel": [RBF(l) for l in np.logspace(-2, 2, 10)]}
kr = GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)
#kr = KernelRidge(alpha=1.0)
stime = time.time()
kr.fit(X0_train, y0_train, sample_weight = W0)
print("Time for KRR fitting: %.3f" % (time.time() - stime))
print('training result: %f'%r2_score(y0_train, kr.predict(X0_train)))
print('test result: %f'%r2_score(y0_test, kr.predict(X0_test)))
'''
