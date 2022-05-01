
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_trian = X_train.reshape(60_000,28,28,1)
X_test = X_test.astype('float32')/255

##convert = lambda i : [ 1.0 if i == x else 0.0 for x in range(10)]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

cnn = Sequential()
cnn.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', input_shape = (28,28,1)))
cnn.add(MaxPool2D(pool_size = (2,2)))





