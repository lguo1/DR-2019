from exp_objects import *
import time
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel as W, ConstantKernel as C, Matern, RBF, ExpSineSquared as ESS, RationalQuadratic as RQ, DotProduct as DP
from sklearn.metrics import r2_score

np.random.seed(0)
GP_0 = pd.read_csv("./saves/GP_0.csv", index_col=0)
print('data\n%s'%(GP_0.tail()))
features_0 = GP_0[-1000:][['infoset','action0','action1']]
outputs_0 = GP_0[-1000:][['v_a0','v_a1','v_a2']]
X0_train, X0_test, y0_train, y0_test = train_test_split(features_0, outputs_0)

# kernel = W() + C(10, (1e-3, 1e1)) + C(1, (1e-3, 1e1)) * RBF(10, (1e-3, 1e1)) + C(1, (1e-3, 1e1)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e1), nu=1.5) + C(1, (1e-3, 1e1)) * RationalQuadratic(length_scale=1.0, alpha=0.1)
# kernel = W() + C(1, (1e-3, 1e1)) + C(1, (1e-3, 1e1)) * RBF(1, (1e-3, 1e1)) + C(1, (1e-3, 1e1)) * ESS(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + C(1, (1e-3, 1e1)) * DP() + C(1, (1e-3, 1e1)) * ESS(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) * RBF(1, (1e-3, 1e1)) + C(1, (1e-3, 1e1)) * ESS(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) * DP() + C(1, (1e-3, 1e1)) * RBF(1, (1e-3, 1e1)) * DP() + C(1, (1e-3, 1e1)) * ESS(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) * RBF(1, (1e-3, 1e1)) * DP()
kernel = W() + C(1, (1e-3, 1e1)) + C(1, (1e-3, 1e1)) * RBF(1, (1e-3, 1e1)) + ESS(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) * DP()
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
stime = time.time()
gp.fit(X0_train, y0_train)
print('\nplayer 0')
print("Posterior (kernel: %s)\n Log-Likelihood: %.3f" % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)))
print('training result: %f'%r2_score(y0_train, gp.predict(X0_train)))
print('test result: %f'%r2_score(y0_test, gp.predict(X0_test)))
print("Time for GPR fitting: %.3f\n" % (time.time() - stime))
