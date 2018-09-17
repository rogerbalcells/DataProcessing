# DataProcessing

import numpy as np
import scipy.sparse as sp


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

a = np.matrix(sample_spherical(1024))
def data(A, m,n, den = 0.01):
    B = []
    for i in range(n):
        AA = A[:]
        x = sp.random(n, 1, density=den, format='csc',
                      data_rvs=lambda s: np.random.uniform(-1, 1, size=s))
        for i in range(len(x)):
            if x[i]!=0:
                x = 1
        e = np.random.rand(m)
        AA = np.mat(AA)*x
        for i in range(m):
            AA[i][0] += e[i]*2-1
        B.append(AA)
    return B

B = data(a,3, 1024, 0.01)
print(B)
print(len(B))


cur_x = 3 # The algorithm starts at x=3
rate = 0.01 # Learning rate
precision = 0.000001 #This tells us when to stop the algorithm
previous_step_size = 1 #
max_iters = 10000 # maximum number of iterations
iters = 0 #iteration counter
df = lambda x: 2*(x+5) #Gradient of our function
while previous_step_size > precision and iters < max_iters:
    prev_x = cur_x #Store current x value in prev_x
    cur_x = cur_x - rate * df(prev_x) #Grad descent
    previous_step_size = abs(cur_x - prev_x) #Change in x
    iters = iters+1 #iteration count
    print("Iteration",iters,"\nX value is",cur_x) #Print iterations

print("The local minimum occurs at", cur_x)
