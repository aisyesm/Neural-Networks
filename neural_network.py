from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import Callback
from keras import backend as K
from keras import optimizers, regularizers

# input image dimensions
img_rows, img_cols = 28, 28

num_classes = 10
batch_size = 32
epochs = 50
adam = optimizers.Adam(lr = 0.0001)

testLosses = []
testAccurracies = []

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
    
    def on_epoch_end(self, epoch, logs = {}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        testLosses.append(loss)
        testAccurracies.append(acc)
        print('\ntest_loss: {}, tets_acc: {}\n'.format(loss, acc))

def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget
    
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest
    
x_train, x_valid, x_test, y_train, y_valid, y_test = loadData()
y_train, y_valid, y_test = convertOneHot(y_train, y_valid, y_test)   

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_valid = x_valid.reshape(x_valid.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
    axis  = 1
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    axis  = -1

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_valid /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')


# Initialising the CNN
classifier = Sequential()

classifier.add(Convolution2D(32, 3, input_shape = input_shape, padding = 'same', activation = 'relu'))

classifier.add(BatchNormalization(axis = axis, momentum = 0.99))

classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 784, activation = 'relu'))

#classifier.add(Dropout(0.5))  used for Dropout section

classifier.add(Dense(output_dim = num_classes, activation = 'softmax'))

classifier.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])

history  = classifier.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=[TestCallback((x_test, y_test))],
              validation_data=(x_valid, y_valid), 
              shuffle = True)
 

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(testLosses)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation', 'Test'], loc='upper right')
plt.show()

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(testAccurracies)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation', 'Test'], loc='lower right')
plt.show()
