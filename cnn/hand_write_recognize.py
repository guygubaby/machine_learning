import numpy as np
from keras.layers import Convolution2D,MaxPool2D,Flatten,Dense
from keras.activations import relu,softmax
from keras.models import Sequential
from keras.optimizers import SGD,Adagrad,Adam
from keras.losses import mse
from keras.utils import np_utils
from keras.preprocessing import image

number = 10000

# https://www.cnblogs.com/tiandsp/p/9638876.html

def load_data(path='../nn/data/mnist.npz'):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    # (x_train, y_train), (x_test, y_test)=mnist.load_data() # can't download data


    x_train=x_train[0:number]
    y_train=y_train[0:number]

    # img=Image.fromarray(x_train[0])
    # img.show()

    x_train=x_train.reshape(number,28,28)

    x_test=x_test.reshape(x_test.shape[0],28,28)

    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')

    y_train=np_utils.to_categorical(y_train,10)
    y_test=np_utils.to_categorical(y_test,10)

    x_train=x_train/255
    x_test=x_test/255

    # add noise
    # x_test=np.random.normal(x_test)
    print(x_train[0].reshape(28,28,1).shape)

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':

    batch_size=100
    epochs=10
    (x_train_temp,y_train_temp),(x_test,y_test)=load_data(path='../nn/data/mnist.npz')

    x_train=[]
    y_train=[]

    for i in range(number):
        x_train.append(x_train_temp[i].reshape(28,28,1))
        y_train.append(y_train_temp[i].reshape(10,1))

    x_train=np.array(x_train)
    y_train=np.array(y_train)

    print(y_train[0].shape)

    # y_train=y_train_temp
    model=Sequential()
    # 1*28*28
    model.add(Convolution2D(25,kernel_size=(3,3),activation=relu,input_shape=(28,28,1)))
    # 25*26*26
    model.add(MaxPool2D((2,2)))
    # 25*13*13
    model.add(Convolution2D(50,kernel_size=(3,3),activation=relu))
    # 50*11*11
    model.add(MaxPool2D((2,2)))
    # 50*5*5

    model.add(Flatten())
    # vector of 1250

    model.add(Dense(units=50*5*5,activation=relu))
    model.add(Dense(units=81,activation=relu))
    model.add(Dense(units=10,activation=softmax))

    model.summary()

    model.compile(optimizer=Adagrad(lr=0.01),loss=mse,metrics=['accuracy'])

    model.fit(x_train[0],y_train[0],batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test))

    score=model.evaluate(x_test,y_test)

    res=np.argmax(model.predict(x_test),axis=1)

    print(res)

    print('test loss : {} , acc : {}'.format(score[0],score[1]))


