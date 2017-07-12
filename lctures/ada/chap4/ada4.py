import numpy as np
import matplotlib.pyplot as plt

def l_turkey(r,meu):
	if(np.absolute(r) <= meu):
		return (1-(1-r**2/meu**2)**3)/6
	else:
		return 1/6

def w_turkey(r,meu):
	if(np.absolute(r) <= meu):
		return (1-r**2/meu**2)**2
	else:
		return 0

def loss (th,x,y):
	return th[0]+th[1]*x-y

def th_up(w,x,y):
	fi = np.append(np.ones(10)[:,np.newaxis],x[:,np.newaxis],axis=1)
	return np.linalg.inv(fi.T.dot(w).dot(fi)).dot(fi.T.dot(w).dot(y))

def main():
	meu = 1.0
	np.random.seed(0)
	n = 10
	N = 1000
	x = np.linspace(-3,3,n)
	y = x+0.2*np.random.randn(n)
	y[n-1] = -4
	y[n-2] = -4
	y[1] = -4

	X = np.linspace(-3,3,N)

	th = np.ones(2)
	w = np.zeros(n*n).reshape(n,n)

	for i in range(1000):
		r = th[0]+th[1]*x-y
		for j in range(n):
			w[j,j] = w_turkey(r[j],meu)
		th = th_up(w,x,y)

	Y = th[0]+th[1]*X

	plt.title("turkey")
	plt.plot(x,y,'ro')
	plt.plot(X,Y,'b-')
	plt.savefig("ada4.png")

if __name__ == '__main__':
	main()
