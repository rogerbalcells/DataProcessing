import numpy as np
import scipy.sparse as sp

class createData:
    def __init__(self, m, n):
        ###Initializing the class that creates the dataset and solves for D
        self.n = n
        self.size = size
        self.a = np.matrix(self.sample_spherical(n))
        self.B = self.data(self.a, self.m, self.n,0.01)
    def sample_spherical(npoints, ndim=3): ###creates a random matrix of the given size(npoints ndim)
        vec = np.random.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0) ###forces the matrice's columns to have a length of 1
        return vec
    def data(self, A, m, p, den = 0.01):
        ## Creates the matrix B = Aixi+E
        B = []
        n = len(A) ###check!!!!!
        for i in range(p):
            AA = A[:]
            x = sp.random(n, 1, density=den, format='csc',
                          data_rvs=lambda s: np.random.uniform(-1, 1, size=s)) #creating a sparse vector x
            for i in range(len(x)): #making the non-zero values of x to become 1
                if x[i]!=0:
                    x = 1
            e = np.random.rand(m) # creating a vector of noise E
            AA = np.mat(AA)*x #multiplying  A*x
            for i in range(len(e)):   # adding noise E to A*x
                AA[i][0] += e[i]*2-1
            B.append(AA)  # adding the result bi to the matrix B
        return B
        def f(self):
            aa = np.matrix(self.sample_spherical(n))
            x = sp.random(n, 1, density=den, format='csc',
                          data_rvs=lambda s: np.random.uniform(-1, 1, size=s))
            xl = 0
            for i in range(len(x)):
                if x != 0:
                    xl += 1
                    x[i] = 1
            res = 0
            for k in range(len(self.B)):
                temp1 = 0
                temp2 = 0
                for j in range(len(self.B[k])):
                    temp1 += aa[k][j]*x[j]-b[k]
                for i in range(len(self.B[k])):
                    temp2 += aa[k][i]*x[i]-b[i]
                res += temp1*temp2



            rate = 0.01 # Learning rate
            precision = 0.000001 #This tells us when to stop the algorithm
            previous_step_size = 1 #
            max_iters = 10000 # maximum number of iterations
            iters = 0 #iteration counter
            df = lambda x: 2*(x+5) #Gradient of our function
            while previous_step_size > precision and iters < max_iters:
                prev_aa = aa #Store current x value in prev_x
                aa = aa - rate * df(prev_aa) #Grad descent
                previous_step_size = abs(aa - prev_aa) #Change in x
                iters = iters+1 #iteration count
                print("Iteration",iters,"\nX value is",aa) #Print iterations
            return aa

B = data(a,3, 100, 0.01)
print(B)
print(len(B))
