import os  
import numpy as np 
import cv2 
import os
import numpy as np
from tensorflow.keras.utils import to_categorical

from keras.layers import Input, Dense 
from keras.models import Model

is_init = False
size = -1

label = []
dictionary = {}
c = 0

for i in os.listdir():
	if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
		if not(is_init):
			is_init = True 
			X = np.load(i)
			size = X.shape[0]
			y = np.array([i.split('.')[0]]*size).reshape(-1,1)
		else:
			X = np.concatenate((X, np.load(i)))
			y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))

		label.append(i.split('.')[0])
		dictionary[i.split('.')[0]] = c  
		c = c+1


for i in range(y.shape[0]):
	y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

###  hello = 0 nope = 1 ---> [1,0] ... [0,1]

y = to_categorical(y)

X_new = X.copy()
y_new = y.copy()
counter = 0 

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt: 
	X_new[counter] = X[i]
	y_new[counter] = y[i]
	counter = counter + 1


ip = Input(shape=(X.shape[1]))

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)

op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

model.fit(X, y, epochs=50)


from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Initialize variables
is_initialized = False
label_list = []
label_dict = {}

# Load the data and labels
for filename in os.listdir():
    if filename.endswith(".npy") and not filename.startswith("labels"):
        data = np.load(filename)
        label_name = filename.split('.')[0]

        if not is_initialized:
            X = data
            y = np.array([label_name] * data.shape[0]).reshape(-1, 1)
            is_initialized = True
        else:
            X = np.concatenate((X, data))
            y = np.concatenate((y, np.array([label_name] * data.shape[0]).reshape(-1, 1)))

        label_list.append(label_name)
        label_dict[label_name] = len(label_dict)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.ravel())
y_one_hot = to_categorical(y_encoded)

# Shuffle and split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.1, random_state=42)

# Define the model
model = Sequential()
model.add(Input(shape=(X.shape[1],)))
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(y_one_hot.shape[1], activation="softmax"))

# Compile the model
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Save the model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))
np.save("labels.npy", label_encoder.classes_)

# Print the label dictionary for reference
print("Label Dictionary:", label_dict)