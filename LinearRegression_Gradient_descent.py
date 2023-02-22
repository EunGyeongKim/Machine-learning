# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# 1D
#create data
np.random.seed(seed=1)
x_min = 4
x_max = 30
x_n = 16
x = 5+25 * np.random.rand(x_n)
prm_c = [170, 108, 0.2] # 생성 매개변수
t = prm_c[0] - prm_c[1] * np.exp(-prm_c[2]*x) + 4 *np.random.randn(x_n)

plt.figure(figsize = (4,4))
plt.plot(x, t, marker='o', linestyle ='None', markeredgecolor='black', 
         color = 'cornflowerblue')
plt.xlim(x_min, x_max)
plt.grid(True)
plt.show()

from mpl_toolkits.mplot3d import Axes3D

# mse 함수
def mse_line(x, t, w):
    y = w[0] * x + w[1]
    mse = np.mean((y-t)**2)
    return mse

# calculation
xn = 100 # 등고선 표시 해상도
w0_range=[-25, 25]
w1_range=[120, 170]
x0 = np.linspace(w0_range[0], w0_range[1], xn)
x1 = np.linspace(w1_range[0], w1_range[1], xn)
xx0, xx1 = np.meshgrid(x0, x1)
j = np.zeros((len(x0), len(x1)))

for i0 in range(xn):
    for i1 in range(xn):
        j[i1, i0] = mse_line(x, t, (x0[i0], x1[i1]))

#graph
plt.figure(figsize = (9.5, 4))
plt.subplots_adjust(wspace = 0.5)

ax = plt.subplot(1, 2, 1, projection='3d')
ax.plot_surface(xx0, xx1, j, rstride=10, cstride=10, alpha=0.3, color='blue', edgecolor='black')
ax.set_xticks([-20, 0, 20])
ax.set_yticks([120, 140, 160])
ax.view_init(20, -60)

plt.subplot(1,2,2)
cont = plt.contour(xx0, xx1, j, 30, colors='black', levels=[100, 1000, 10000, 100000])
cont.clabel(fmt = '%1.0f', fontsize =8)
plt.grid(True)
plt.show()

#평균제곱 오차의 기울기
def dmse_line(x, t, w):
    y = w[0]*x+w[1]
    d_w0 = 2*np.mean((y-t)*x)
    d_w1 = 2*np.mean(y-t)
    return d_w0, d_w1

# 경사하강법
def fit_line_num(x, t):
    w_init = [10.0, 165.0]
    alpha = 0.001 # learning rate
    i_max = 100000 # max iter 
    eps = 0.1 # 반복 종료 기울기의 절대값의 한계
    w_i = np.zeros([i_max, 2])
    w_i[0, :] = w_init
    for i in range(1, i_max):
        dmse = dmse_line(x, t, w_i[i-1])
        w_i[i, 0] = w_i[i-1, 0]-alpha * dmse[0]
        w_i[i, 1] = w_i[i-1, 1]-alpha * dmse[1]
        if max(np.absolute(dmse)) < eps:
            # 종료판정. 
            break
    w0 = w_i[i,0]
    w1 = w_i[i,1]
    w_i = w_i[:i, :]
    return w0, w1, dmse, w_i

# main
plt.figure(figsize=(4,4))
xn = 100 #등고선 해상도
w0_range = [-25, 25]
w1_range = [120, 170]
w0 = np.linspace(w0_range[0], w0_range[1], xn)
w1 = np.linspace(w1_range[0], w1_range[1], xn)
xx0, xx1 = np.meshgrid(x0, x1)
j = np.zeros((len(x0), len(x1)))
for i0 in range(xn):
    for i1 in range(xn):
        j[i1, i0] = mse_line(x, t, (x0[i0], x1[i1]))
cont = plt.contour(xx0, xx1, j, 30, colors='black', levels=(100, 1000, 10000, 100000))
cont.clabel(fmt='%1.0f', fontsize = 8)
plt.grid(True)

# 경사하강법 호출
W0, W1, dMSE, W_history = fit_line_num(x, t)

# result
print('반복횟수 {0}'.format(W_history.shape[0]))
print('W=[{0: .6f}. {1:6f}]'.format(dMSE[0], dMSE[1]))
print('MSE={0:.6f}'.format(mse_line(x, t, [W0, W1])))
plt.plot(W_history[:,0], W_history[:,1], '.-', color = 'gray', markersize = 10, markeredgecolor='cornflowerblue')
plt.show()

def show_line(w):
    xb = np.linspace(x_min, x_max, 100)
    y = w[0] * xb + w[1]
    plt.plot(xb, y, color=(.5, .5, .5), linewidth=4)

plt.figure(figsize=(4,4))
W = np.array([W0, W1])
mse = mse_line(x, t, W)
print('w0={0:.3f}, w1 = {1:.3f}'.format(W0, W1))
print('SD = {0:.3f} cm'.format(np.sqrt(mse)))
show_line(W)
plt.plot(x, t, marker='o', linestyle = 'None', color='cornflowerblue', markeredgecolor='black')
plt.xlim(x_min, x_max)
plt.grid(True)
plt.show()

# 해석해
def fit_line(x, t):
    mx = np.mean(x)
    mt = np.mean(t)
    mtx = np.mean(t*x)
    mxx = np.mean(x*x)
    w0 = (mtx - mt * mx) / (mxx - mx**2)
    w1 = mt - w0 * mx
    return np.array([w0, w1])

# main
W = fit_line(x, t)
print("w0={0:.3f}, w1={1:.3f}".format(W[0], W[1]))
mse = mse_line(x, t, W)
print("SD={0:.3f}cm".format(np.sqrt(mse)))
plt.figure(figsize = (4,4))
show_line(W)
plt.plot(x, t, marker='o', linestyle='None', color='cornflowerblue', markeredgecolor = 'black')
plt.xlim(x_min, x_max)
plt.grid(True)
plt.show()







