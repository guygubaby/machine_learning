from keras.models import load_model
from keras.utils import np_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def get_model():
    path='../saved_models/hand_write_10_epoches.h5'
    model=load_model(path)
    return model

def get_max_numpy_index(np_arr,index):
    position=np.unravel_index(np_arr.argmax(),np_arr.shape)
    return position[index]


if __name__ == '__main__':
    model=get_model()
    number=200

    with np.load('../data/mnist.npz') as f:
        x_test,y_test=f['x_test'],f['y_test']

    x_test=x_test[0:number]
    y_test=y_test[0:number]

    x_test=x_test.astype('float32')

    y_test=np_utils.to_categorical(y_test,10)

    # img=Image.fromarray(x_test[0])
    # img.show()

    predictions=[]
    truth=[]

    for i in range(0,number):
        x=x_test[i].reshape(1,28*28)/255
        y=y_test[i]

        res=model.predict(x)
        predict_number = get_max_numpy_index(res,1)
        true_number=get_max_numpy_index(y,0)
        predictions.append(predict_number)
        truth.append(true_number)

    data={
        'prediction':predictions,
        'truth':truth
    }

    plt.figure()
    plt.plot(predictions,'rs',truth,'bs')
    plt.show()

    right_count=(np.asarray(predictions)==np.asarray(truth)).sum()
    percent=right_count/number

    print('acc is : {} %'.format(percent*100))