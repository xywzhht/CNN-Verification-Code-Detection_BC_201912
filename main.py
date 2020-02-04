#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
import pandas as pd
import string
import cv2 as cv
import numpy as np

data_path = 'C:/Users/kt1212/Desktop/train/train_label.csv'
result_path= 'C:/Users/kt1212/Desktop/submission.csv'
train_img_path="C:/Users/kt1212/Desktop/train"
test_img_path="C:/Users/kt1212/Desktop/test"

characters = string.digits + string.ascii_uppercase + string.ascii_lowercase
print(characters)

width, height, n_len, n_class = 120, 40, 4, len(characters)

# numpy.argmax 返回数组最大值的索引
# axis表示纬度，默认为0，返回沿轴axis最大值的索引
# one hot解码
def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

class Dataset:
    def __init__(self,data_path):
        self.x_train=[]
        self.y_train=[]
        self.x_val=[]
        self.y_val=[]
        # 数据集加载路径
        self.path_name = data_path

    def load(self):
        data = pd.read_csv(self.path_name, encoding="utf-8")
        self.y_train = [np.zeros((len(data), n_class), dtype=np.uint8) for i in range(n_len)]
        # len(data)==5000
        # range(n)==[0,n)
        for i in range(len(data)):
            self.x_train.append(cv.bilateralFilter(cv.imread(train_img_path+'/'+data['ID'][i]), 3, 560, 560))
            label = data['label'][i]
            # j表示序号（从0开始），ch表示label[j]的字母
            for j, ch in enumerate(label):
                self.y_train[j][i, :] = 0
                self.y_train[j][i, characters.find(ch)] = 1
        self.x_train=np.array(self.x_train)


class CNN_Model:
    def __init__(self):
        self.model=None

    def build_model(self):
        input_tensor = Input((height, width, 3))
        x = input_tensor

        for i, n_cnn in enumerate([2, 2, 2, 2, 2]):
            # for j in range(2):
            for j in range(1):
                x = Conv2D(32 * 2 ** min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
            x = MaxPooling2D(2)(x)

        x = Flatten()(x)
        x = [Dense(n_class, activation='softmax', name='c%d' % (i + 1))(x) for i in range(n_len)]

        self.model = Model(input=input_tensor, output=x)

    def train(self,x_train,y_train,x_val=None,y_val=None):
        # 随迭代过程更新，cnn_captcha_break.h5结束后才更新
        callbacks = [ModelCheckpoint('C:/Users/kt1212/Desktop/cnn_best.h5')]
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(1e-7, amsgrad=True),
                      metrics=['accuracy'])
        self.model.fit(x_train,y_train,epochs=1, validation_data=None, callbacks=callbacks)

    def save_model(self,file_path):
        self.model.save(file_path)

    def load_model(self,file_path):
        self.model = load_model(file_path)

    def predict(self,img_path):
        img = cv.imread(img_path)
        result = self.model.predict(np.array([cv.bilateralFilter(img, 3, 560, 560)]))
        return decode(result)

    def test(self):
        test_label = pd.read_csv(result_path)
        for i in range(len(test_label)):
            res = self.predict(test_img_path+'/'+test_label['ID'][i])
            test_label['label'][i]=res
            print(i)
        test_label.to_csv(result_path,index=False,encoding="UTF-8")


if __name__ == '__main__':

    #train
    dataset = Dataset(data_path)
    dataset.load()

    model = CNN_Model()
    model.load_model("C:/Users/kt1212/Desktop/cnn_captcha_break.h5")  # 加载上一次训练好的模型
    # model.build_model()
    model.train(dataset.x_train,dataset.y_train)
    model.save_model("C:/Users/kt1212/Desktop/cnn_captcha_break.h5")         # 保存最终模型

    # #test 生成结果文件
    # model = CNN_Model()
    # model.load_model("C:/Users/kt1212/Desktop/cnn_captcha_break.h5")
    # model.test()
