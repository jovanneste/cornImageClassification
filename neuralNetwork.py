import cv2
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

train_data = []
train_labels = []

df = pd.read_csv('corn/train.csv')
#only first 100 rows
df=df.head(100)

def labelToNum(label):
	if (label=='pure'): return 0
	elif (label=='broken'): return 1
	elif (label=='discolored'): return 2
	else: return 3

for index, row in df.iterrows():
	id = row.iloc[0]
	img = row.iloc[2]
	label = row.iloc[3]

	image_array = cv2.imread("corn/"+img)
	image_array = cv2.resize(image_array, (64,64))

	train_data.append(image_array)
	train_labels.append(labelToNum(label))



train_data = np.array(train_data)
train_data=train_data/255.0

print(train_data.shape) #prints out (100, 64, 64, 3) might need to flatten to (100, 64*64)


onehot_target = pd.get_dummies(train_labels)
x_train, x_test, y_train, y_test = train_test_split(train_data, onehot_target, test_size=0.1, random_state=20)

def sigmoid(s):
    return 1/(1+np.exp(-s))

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def sigmoid_derv(s):
    return s*(1-s)

def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def loss(pred, real):
	n_samples = real.shape[0]
	logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
	loss = np.sum(logp)/n_samples
	return loss


class NeuralNetwork:
	def __init__(self, x, y):
		self.x = x
		neurons = 128
		self.lr = 0.5
		input_dim = x.shape[1]
		output_dim = y.shape[1]
		self.w1 = np.random.randn(input_dim, neurons)
		self.w2 = np.random.randn(neurons, neurons)
		self.w3 = np.random.randn(neurons, output_dim)
		self.b1 = np.zeros((1, neurons))
		self.b2 = np.zeros((1, neurons))
		self.b3 = np.zeros((1, output_dim))
		self.y = y

	def feedforward(self):
		z1 = np.dot(self.x, self.w1) + self.b1
		self.a1 = sigmoid(z1)
		z2 = np.dot(self.a1, self.w2) + self.b2
		self.a2 = sigmoid(z2)
		z3 = np.dot(self.a2, self.w3) + self.b3
		self.a3 = softmax(z3)

	def backprop(self, x):
		if (x%100==0):
			print('Error:', loss(self.a3, self.y))
		a3_delta = cross_entropy(self.a3, self.y)
		z2_delta = np.dot(a3_delta, self.w3.T)
		a2_delta = z2_delta * sigmoid_derv(self.a2)
		z1_delta = np.dot(a2_delta, self.w2.T)
		a1_delta = z1_delta * sigmoid_derv(self.a1)

        #adjusting weights and biases
		self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
		self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
		self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
		self.b2 -= self.lr * np.sum(a2_delta, axis=0)
		self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
		self.b1 -= self.lr * np.sum(a1_delta, axis=0)
