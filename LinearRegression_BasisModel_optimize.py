# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# data 
np.random.seed(seed =1)
x_min = 4
x_max = 30
x_n = 16
x = 5+25 * np.random.rand(x_n)
prm_c = [170, 108, 0.2]
t = prm_c[0] - prm_c[1] * np.exp(-prm_c[2] *x) + 4*np.random.randn(x_n)

# Gaussian
def gauss(x, mu, s):
    return np.exp(-(x-mu)**2/(2*s**2))

# main
m = 4
plt.figure(figsize = (4,4))
mu = np.linspace(5, 30, m)
s = mu[1] - mu[0]
xb = np.linspace(x_min, x_max, 100)
for j in range(m):
    y = gauss(xb, mu[j], s)
    plt.plot(xb, y, color='gray', linewidth=3)
plt.grid(True)
plt.xlim(x_min, x_max)
plt.ylim(0, 1.2)
plt.show()

# linear basis function model
def gauss_func(w, x):
    m = len(w)-1
    mu = np.linspace(5, 30, m)
    s = mu[1] - mu[0]
    # make y array with same size x
    y = np.zeros_like(x)
    for j in range(m):
        y = y + w[j] * gauss(x, mu[j], s)
    y = y+w[m]
    return y

# linear basis modeul MSE
def mse_gauss_func(x, t, w):
    y = gauss_func(w, x)
    mse = np.mean((y-t)**2)
    return mse

# exact solution of a linear basis function model
def fit_gauss_func(x, t, m):
    mu = np.linspace(5, 30, m)
    s= mu[1] - mu[0]
    n = x.shape[0]
    psi = np.ones((n, m+1))
    for j in range(m):
        psi[:, j] = gauss(x, mu[j], s)
    psi_t = np.transpose(psi)

    # 선형대수 함수 (=np.linalg.in)
    b = np.linalg.inv(psi_t.dot(psi))
    c = b.dot(psi_t)
    w = c.dot(t)
    return w

# show gauss function
def show_gauss_func(w):
    xb = np.linspace(x_min, x_max, 100)
    y = gauss_func(w, xb)
    # lw =  line width
    plt.plot(xb, y, c=[.5, .5, .5], lw = 4)

# main
plt.figure(figsize =(4,4))
m =4
w = fit_gauss_func(x, t, m)
show_gauss_func(w)
plt.plot(x, t, marker='o', linestyle='None', color='cornflowerblue', markeredgecolor='black')
plt.xlim(x_min, x_max)
plt.grid(True)
mse = mse_gauss_func(x, t, w)
print('w ='+str(np.round(w, 1)))
print("SD={0:.3f}".format(np.sqrt(mse)))
plt.show()

plt.figure(figsize =(10, 2.5))
plt.subplots_adjust(wspace = 0.3)
m = [2, 4, 7, 9]
for i in range(len(m)):
    plt.subplot(1, len(m), i+1)
    w = fit_gauss_func(x, t, m[i])
    show_gauss_func(w)
    plt.plot(x, t, marker ='o', linestyle = "None", color='cornflowerblue', markeredgecolor='black')
    plt.xlim(x_min, x_max)
    plt.grid(True)
    plt.ylim(130, 180)
    mse = mse_gauss_func(x, t, w)


    plt.title("m={0:d}, SD={1:.1f}".format(m[i], np.sqrt(mse)))
plt.show()

# train & test data
x_test =x[:int(x_n/4+1)]
t_test =t[:int(x_n/4+1)]
x_train =x[int(x_n/4+1):]
t_train =t[int(x_n/4+1):]

# main
plt.figure(figsize=(10, 2.5))
plt.subplots_adjust(wspace=0.3)
m = [2,4,7,9]
for i in range(len(m)):
    plt.subplot(1, len(m), i+1)
    w = fit_gauss_func(x_train, t_train, m[i])
    show_gauss_func(w)
    plt.plot(x_train, t_train, marker='o', linestyle='None', color='white', markeredgecolor='black', label='trainig')
    plt.plot(x_test, t_test, marker='o', linestyle='None', color='cornflowerblue', markeredgecolor='black', label='test')
    plt.legend(loc='lower right', fontsize = 10, numpoints=1)
    plt.xlim(x_min, x_max)
    plt.ylim(130, 180)
    plt.grid(True)
    mse = mse_gauss_func(x_test, t_test, w)
    
    plt.title("m={0:d}, SD={1:.1f}".format(m[i], np.sqrt(mse)))
plt.show()

plt.figure(figsize = (5,4))
m = range(2, 10)
mse_train = np.zeros(len(m))
mse_test = np.zeros(len(m))
for i in range(len(m)):
    w = fit_gauss_func(x_train, t_train, m[i])
    mse_train[i] = np.sqrt(mse_gauss_func(x_train, t_train, w))
    mse_test[i] = np.sqrt(mse_gauss_func(x_test, t_test, w))
plt.plot(m, mse_train,  marker='o', linestyle='-', color='white', markeredgecolor='black', label='trainig')
plt.plot(m, mse_test,  marker='o', linestyle='-', color='cornflowerblue', markeredgecolor='black', label='test')
plt.legend(loc='lower right', fontsize = 10)
plt.show()

# k fold cross validation
def kfold_gauss_func(x, t, m, k):
    n= x.shape[0]
    mse_train = np.zeros(k)
    mse_test = np.zeros(k)

    for i in range(0, k):
        # fmod = n을 k로 나눈 나머지를 출력함
        x_train = x[np.fmod(range(n), k) != i]
        t_train = t[np.fmod(range(n), k) != i]
        x_test = x[np.fmod(range(n), k) == i]
        t_test = t[np.fmod(range(n), k) == i]

        wm = fit_gauss_func(x_train, t_train, m)
        mse_train[i] = mse_gauss_func(x_train, t_train, wm)
        mse_test[i] = mse_gauss_func(x_test, t_test, wm)
    return mse_train, mse_test

# main
m = range(2, 8)
k = 16
cv_gauss_train = np.zeros((k, len(m)))
cv_gauss_test = np.zeros((k, len(m)))

for i in range(0, len(m)):
    cv_gauss_train[:, i], cv_gauss_test[:, i] = kfold_gauss_func(x, t, m[i], k)

mean_gauss_train = np.sqrt(np.mean(cv_gauss_train, axis = 0))
mean_gauss_test = np.sqrt(np.mean(cv_gauss_test, axis = 0))

plt.figure(figsize = (4, 3))
plt.plot(m, mean_gauss_train, marker = 'o', linestyle='-', color='white', markeredgecolor='black', label='trainig')
plt.plot(m, mean_gauss_test, marker = 'o', linestyle='-', color='cornflowerblue', markeredgecolor='black', label='test')
plt.legend(loc = 'upper left', fontsize=10)
plt.ylim(0, 20)
plt.grid(True)
plt.show()

# find parameter using scipy 
def model(x, w):
    y = w[0] - w[1] * np.exp(-w[2]*x)
    return y

def show_model(w):
    xb = np.linspace(x_min, x_max, 100)
    y = model(xb, w)
    plt.plot(xb, y, c =[.5, .5, .5], lw = 4)

def mse_model(w, x, t):
    y = model(x, w)
    mse = np.mean((y-t)**2)
    return mse

from scipy.optimize import minimize
def fit_model(w_init, x, t):
    res1 = minimize(mse_model, w_init, args=(x, t), method = 'powell')
    return res1.x

# main
plt.figure(figsize=(4,4))
w_init= [100, 0, 0]
w = fit_model(w_init, x, t)

print("w0={0:.3f}, w1={1:.3f}, w2={2:.3f}".format(w[0], w[1], w[2]))
show_model(w)

plt.plot(x, t, marker = 'o', linestyle='None', color='cornflowerblue', markeredgecolor='black')
plt.xlim(x_min, x_max)
plt.grid(True)
mse = mse_model(w, x, t)
plt.title("SD={0:.1f}cm".format(np.sqrt(mse)))
plt.show()