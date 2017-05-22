import numpy as np
import matplotlib.pyplot as plt

def main():
	np.random.seed(0)
	n = 50
	x = np.linspace(-3,3,n)
	pix = np.pi*x
	y = np.sin(pix)/pix+0.1*x+0.2*np.random.randn(n)
	h = 0.3
	l = 0.1
	hh = 2*h**2
	k = np.exp(-(np.square(x-x[:,np.newaxis]))/hh)
	
	#初期値
	th = np.random.randn(k.shape[0])
	z = np.random.randn(k.shape[0])
	u = np.random.randn(k.shape[0])

	num = 100
	error = np.zeros(num)

	X = np.linspace(-3,3,n)
	piX = np.pi * X
	Y = np.sin(piX)/piX+0.1*X
	K = np.exp(-(np.square(X-x[:,np.newaxis]))/hh)

	for i in range(num):
		th = np.linalg.inv(np.dot(k,k)+np.eye(k.shape[0])).dot(k.dot(y)+z-u)
		z = np.maximum(0,th+u-l)+np.minimum(0,th+u+l)
		u = u+th-z

		predy = np.dot(K.T,th)

		error[i] = np.linalg.norm(predy-Y)

	plt.title("hist")
	plt.hist(th)
	plt.savefig("ada3_1.png")
	plt.close()

	plt.title("error")
	plt.plot(np.arange(num),error,'yo-')
	plt.savefig("ada3_2.png")
	plt.close()

	plt.title("result")
	plt.plot(x,y,'go')
	plt.plot(X,predy,'r-')
	plt.savefig("ada3_3.png")

if __name__ == '__main__':
	main()