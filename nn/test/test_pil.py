from PIL import Image
import numpy as np


if __name__ == '__main__':
    img_path='./img/avatar.jpeg'
    img=Image.open(img_path)
    img.show()
    arr=np.asarray(img)
    print(arr.shape)

