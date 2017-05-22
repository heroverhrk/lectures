import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import os

def load(name):
    x = np.ndarray(shape=(0, 256))
    y = np.ndarray(shape=(0, 10))
    for i in range(10):
        _x = pd.read_csv("digit_{}{}.csv".format(name, i), header=None).as_matrix()
        x = np.concatenate([x, _x])
        _y = -np.ones(shape=(_x.shape[0], 10))
        _y[:, i] = 1
        y = np.concatenate([y, _y])
    return x, y

os.chdir("ada/resource/digit")

hh = 1.0
l = 0.01
print('Loading Training Data')
x, y = load('train')
print('Start Training')
x2 = np.sum(x**2, axis=1)
k = np.exp(-(x2+x2[:,None]-2*x.dot(x.T))//hh)
t = (np.linalg.inv(k.dot(k)+l*np.eye(k.shape[0]))).dot(k).dot(y)
print('Loading Test Data')
test_x, test_y = load('test')
print('Start Test')
test_x2 = np.sum(test_x**2, axis=1)
test_k = np.exp(-(x2+test_x2[:, None]-2*test_x.dot(x.T)//hh))
pred = test_k.dot(t)
pred_y = np.argmax(pred, axis=1)
true_y = np.argmax(test_y, axis=1)

print()
print("f1_score : ",f1_score(true_y,pred_y,average='macro'))
