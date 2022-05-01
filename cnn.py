





from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from numpy import genfromtxt
import json
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000,28,28,1)
X_train = X_train.astype('float32')/255

X_test = X_test.reshape(10000,28,28,1)
X_test = X_test.astype('float32')/255

##convert = lambda i : [ 1.0 if i == x else 0.0 for x in range(10)]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

cnn = Sequential()
cnn.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', input_shape = (28,28,1)))
cnn.add(MaxPool2D(pool_size = (2,2)))
cnn.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
cnn.add(MaxPool2D(pool_size = (2,2)))
cnn.add(Flatten())
cnn.add(Dense(units = 128, activation = 'relu'))
cnn.add(Dense(units = 10, activation = 'softmax'))

cnn.summary()

plot_model(cnn, to_file='convnet.png', show_shapes=True, show_layer_names=True)

cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
cnn.fit(X_train,y_train,epochs=5,batch_size=64,validation_split=.1)

cnn.save('mnist_cnn.h5')
cnn = load_model('mnist_cnn.h5')

predictions = cnn.predict(X_test)
df = pd.DataFrame(predictions)
for index, probability in enumerate(predictions[0]):
    print(f'{index}: {probability: .10%}')

images = X_test.reshape(10000,28,28)
incorrect_predictions = []
for i, (p,e) in enumerate(zip(predictions, y_test)):
    predicted, expected = np.argmax(p), np.argmax(e)
    if predicted != expected:
        incorrect_predictions.append((i, images[i], predicted, expected))

np.savetxt('mnist_predictions', predictions, delimiter = "," , fmt = '%s')

np.savetxt('mnist_y_test', y_test, delimiter = "," , fmt = '%s')
np.savetxt('mnist_X_test', X_test.reshape(10000,784), delimiter = "," , fmt = '%s')

with open('mnist_incorrect_predictions', 'w') as ip_js:
    json.dump(pd.Series(incorrect_predictions).to_json(orient = 'values'), ip_js)

figure,axes = plt.subplots(nrows=4, ncols=6,figsize=(16,12))

for axes,item in zip(axes.ravel(),incorrect_predictions):
    index,image,predicted,expected = item
    axes.imshow(image,cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(f'index: {index}\np: {predicted}; e: {expected}')
