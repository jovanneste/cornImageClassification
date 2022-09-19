import cv2
import os
import numpy as np
import pandas as pd

train_data = []
train_labels = []

df = pd.read_csv('corn/train.csv')
for index, row in df.iterrows():
	id = row.iloc[0]
	img = row.iloc[2]
	label = row.iloc[3]

	image_array = cv2.imread("corn/"+img)
	image_array = cv2.resize(image_array, (224,224))

	train_data.append(image_array)
	train_labels.append(label)



train_data = np.array(train_data)
train_data=train_data/255.0

print(train_data.shape)
