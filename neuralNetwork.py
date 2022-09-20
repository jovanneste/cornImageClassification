import cv2
import os
import numpy as np
import pandas as pd

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop, Adadelta, Adam

from sklearn.model_selection import train_test_split

train_data = []
train_labels = []

df = pd.read_csv('corn/train.csv')
#only first 14000 rows
df=df.head(14000)

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
	image_array = cv2.resize(image_array, (32,32))

	train_data.append(image_array)
	train_labels.append(labelToNum(label))



train_data = np.array(train_data)
train_data=train_data/255.0

train_data = train_data.reshape(14000, 3072)

print(train_data.shape) #prints out (100, 64, 64, 3) might need to flatten to (100, 64*64)


onehot_target = pd.get_dummies(train_labels)
x_train, x_test, y_train, y_test = train_test_split(train_data, onehot_target, test_size=0.1, random_state=20)

print(x_train.shape)

model = Sequential()
model.add(Dense(256, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.summary()

model.compile(optimizer=Adadelta(), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=250, batch_size=1000)

scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
