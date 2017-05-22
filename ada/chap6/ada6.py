import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(123)
n = 200
x = np.zeros([3,200])

x[0,:n//2] = np.random.randn(n//2)-5
x[0,n//2:] = np.random.randn(n//2)+5
x[1] = np.random.randn(n)
x[2] = np.ones(n)

y = np.append(np.ones(n//2),-np.ones(n//2))
y[0:3] = -1
y[n//2:n//2+3] = 1

x[1,0:3] -= 5
x[1,n//2:n//2+3] += 5
x = x.T

alpha = np.zeros(n)
beta = 1.0
e_al = 0.0001
e_be = 0.01
epochs = 1000

for epoch in range(epochs):
    for i in range(n):
        delta = 1 - (y[i] * x[i]).dot(alpha * y * x.T).sum() - beta * y[i] * alpha.dot(y)
        alpha[i] += e_al * delta
    for i in range(n):
        beta += e_be * alpha.dot(y) ** 2 / 2

w = (alpha * y).T.dot(x)

for i in range(n):
    if(y[i]==1):
        plt.plot(x[i,0],x[i,1],'bo')
    else:
        plt.plot(x[i,0],x[i,1],'rx')
seq = np.arange(-10, 10)
plt.plot(seq,-(w[2]+w[0]*seq)/w[1],'k-')

plt.savefig("ada6.pdf")
