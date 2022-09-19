import cv2
import os
import numpy as np

train_data = []

for img in os.listdir("corn/train"):
	image_path = "corn/train"+"/"+img
	image_array = cv2.imread(image_path)
	image_array = cv2.resize(image_array, (224,224))
	train_data.append(image_array)



train_data = np.array(train_data)
train_data=train_data/255.0

print(train_data.shape)
