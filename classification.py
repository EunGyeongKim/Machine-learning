
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

np.random.seed(seed=0) 
X_min = 0
X_max = 2.5
X_n = 30
X_col = ['cornflowerblue', 'gray']
X = np.zeros(X_n) 
T = np.zeros(X_n, dtype=np.uint8) 
Dist_s = [0.4, 0.8] 
Dist_w = [0.8, 1.6] 
Pi = 0.5 

for n in range(X_n):
  wk = np.random.rand()
  T[n] = 0 * (wk < Pi) + 1 * (wk >= Pi)  #(A)
  X[n] = np.random.rand() * Dist_w[ T[n] ] + Dist_s[ T[n] ]

def show_data1(x1,t1):
  K = np.max(t1) + 1
  for k in range(K): #(A)
    plt.plot( x1[t1==k], t1[t1==k], X_col[k], alpha=0.5, linestyle='none', marker='o' ) #(B)
    plt.grid(True)
    plt.ylim(-.5, 1.5)
    plt.xlim(X_min,X_max)
    plt.yticks([0,1])

fig = plt.figure(figsize=(3,3))
show_data1(X, T)
plt.show()

"""6.1.4 로지스틱 회귀 모델"""

def logistic(x,w):
  y = 1 / ( 1  + np.exp( -(w[0] * x + w[1])) )
  return y

def show_logistic(w):
  xb = np.linspace(X_min, X_max, 100)
  y = logistic(xb, w)
  plt.plot(xb, y, color='gray', linewidth=4)

  #결정 경계
  i = np.min( np.where(y > 0.5) ) #(A)
  B = (xb[i - 1] + xb[i]) / 2 #(B)
  plt.plot([B, B], [-.5, 1.5], color='k', linestyle='--')
  plt.grid(True)
  return B

# test
W = [8, -10]
show_logistic(W)

def cee_logistic(w,x,t):
    y = logistic(x,w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t[n] * np.log(y[n]) + ( 1 - t[n]) * np.log(1 - y[n]) )
    cee = cee / X_n
    return cee

# test
W = [1,1]
cee_logistic(W, X, T)

# 평균 교차 엔트로피의 오차 미분
def dcee_logistic(w,x,t):
  y = logistic(x,w)
  dcee = np.zeros(2)
  for n in range(len(y)):
    dcee[0] = dcee[0] + ( y[n] - t[n] ) * x[n] 
    dcee[1] = dcee[1] + ( y[n] - t[n] )
  dcee = dcee / X_n
  return dcee

#경사 하강법에 의한 값.
from scipy.optimize import minimize

def fit_logistic(w_init, x, t):
    res1 = minimize(cee_logistic, w_init, args=(x, t),
                    jac=dcee_logistic, method="CG") # (A)
    return res1.x


# 메인 ------------------------------------
plt.figure(1, figsize=(3, 3))
W_init=[1,-1]
W = fit_logistic(W_init, X, T)
print("w0 = {0:.2f}, w1 = {1:.2f}".format(W[0], W[1]))
B=show_logistic(W)
show_data1(X, T)
plt.ylim(-.5, 1.5)
plt.xlim(X_min, X_max)
cee = cee_logistic(W, X, T)
print("CEE = {0:.2f}".format(cee))
print("Boundary = {0:.2f} g".format(B))
plt.show()



















