import numpy as np
from keras.losses import categorical_crossentropy,mse
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD,Adam,Adagrad
from keras.utils import np_utils
from keras.datasets import mnist
from PIL import Image


def load_data(path='./data/mnist.npz'):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    # (x_train, y_train), (x_test, y_test)=mnist.load_data() # can't download data

    number=10000

    x_train=x_train[0:number]
    y_train=y_train[0:number]

    # img=Image.fromarray(x_train[0])
    # img.show()

    x_train=x_train.reshape(number,28*28)
    x_test=x_test.reshape(x_test.shape[0],28*28)

    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')

    y_train=np_utils.to_categorical(y_train,10)
    y_test=np_utils.to_categorical(y_test,10)

    x_train=x_train/255
    x_test=x_test/255

    # add noise
    # x_test=np.random.normal(x_test)

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    epochs=10
    (x_train, y_train), (x_test, y_test)=load_data()

    # add dropout just like one epoch trained M nnw ， fix overfitting

    model=Sequential()
    model.add(Dense(input_dim=28*28,units=500,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=500,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=10,activation='softmax'))
    model.add(Dropout(0.5))

    model.summary()

    model.compile(loss=mse,optimizer=Adagrad(lr=0.01),metrics=['accuracy'])

    model.fit(x_train,y_train,batch_size=100,epochs=epochs)

    res_train=model.evaluate(x_train,y_train)
    print('train loss : {}, acc : {}'.format(res_train[0],res_train[1]))

    result = model.evaluate(x_test,y_test)
    model.save('./saved_models/hand_write_{}_epoches.h5'.format(epochs))
    print('test loss : {}, acc : {}'.format(result[0],result[1]))

    # 0.005827153577861327 0.9619