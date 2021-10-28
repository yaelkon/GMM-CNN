import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np
from scipy.stats import multivariate_normal

mu_1 = [4,5]
sigma_1 = [[0.5, 0], [0, 0.5]]

x, y = np.random.multivariate_normal(mu_1, sigma_1, 5000).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()


#Parameters to set


mu_1 = [4,5]
mu_1_es = [3.987, 4.987]
sigma_1 = [[0.5, 0], [0, 0.5]]
sigma_1_es = [[0.71, 0], [0, 0.687]]

mu_2 = [2,1]
mu_2_es = [2.019, 0.996]
sigma_2 = [[2, 0], [0, 1]]
sigma_2_es = [[1.402, 0], [0, 1.01]]

mu_3 = [3,3]
mu_3_es = [2.986, 2.968]
sigma_3 = [[1, 0], [0, 1]]
sigma_3_es = [[0.968, 0], [0, 0.992]]

x = np.linspace(-5, 8,500)
y = np.linspace(-5, 8,500)
X,Y = np.meshgrid(x,y)

pos = np.array([X.flatten(),Y.flatten()]).T



rv1 = multivariate_normal(mu_1, sigma_1)
rv2 = multivariate_normal(mu_2, sigma_2)
rv3 = multivariate_normal(mu_3, sigma_3)

rv1_es = multivariate_normal(mu_1_es,sigma_1_es)
rv2_es = multivariate_normal(mu_2_es, sigma_2_es)
rv3_es = multivariate_normal(mu_3_es, sigma_3_es)

fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
ax0.contour(rv1.pdf(pos).reshape(500,500), colors='red')
ax0.contour(rv2.pdf(pos).reshape(500,500), colors='lime')
ax0.contour(rv3.pdf(pos).reshape(500,500), colors='orange')
ax0.contour(rv1_es.pdf(pos).reshape(500,500), colors='royalblue')
ax0.contour(rv2_es.pdf(pos).reshape(500,500), colors='royalblue')
ax0.contour(rv3_es.pdf(pos).reshape(500,500), colors='royalblue')
ax0.set_title('Learning GMM on synthetic data')
# ax0.legend(loc='upper right')
plt.show()
a=0