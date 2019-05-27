import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel as W, ConstantKernel as C, Matern, RBF, ExpSineSquared, RationalQuadratic
from sklearn.metrics import r2_score

np.random.seed(0)
data = pd.read_csv("/home/lguo1/Desktop/train.csv")
# print('data\n%s'%(data.head()))
prices = data['medv']
features = data[['rm', 'lstat', 'ptratio', 'black']]
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=1/3)
#kernel = C(10, (1e-3, 1e3)) + C(1, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + W()
#kernel = C(10, (1e-3, 1e3)) + C(1, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + W()
#kernel = C(10, (1e-3, 1e3)) + C(1, (1e-3, 1e3)) *  ExpSineSquared(length_scale=1.0, periodicity=3.0, length_scale_bounds=(0.1, 10.0), periodicity_bounds=(1.0, 10.0)) + W()
#kernel = C(10, (1e-3, 1e3)) + C(1, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=0.1) + W()
kernel = W() + C(10, (1e-3, 1e3)) + C(1, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + C(1, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
#kernel = W() + C(10, (1e-3, 1e3)) + RBF(10, (1e-3, 1e3)) + Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=1.5)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp.fit(X_train, y_train)
print("Posterior (kernel: %s)\n Log-Likelihood: %.3f" % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)))
print('training result: %f'%r2_score(y_train, gp.predict(X_train)))
print('test result: %f'%r2_score(y_test, gp.predict(X_test)))
