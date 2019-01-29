#-*- coding:utf-8 -*-
import keras
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.applications import InceptionResNetV2,Xception
from keras.layers import GlobalMaxPool2D,Dropout,Dense,MaxPool2D
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard
from keras import Input,Model
import os
from keras.utils.np_utils import to_categorical
from tfrecords_to_dataset import get_data

'消除警告'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def train_model(model,n,epochs,batch_size):

    x_train,y_train = get_data('./../train.tfrecords')
    y_train = to_categorical(y_train, num_classes=119)

    model.summary()

    # if PU
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    result = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                callbacks=[TensorBoard(log_dir='./../result/log/'+n+'_log')])

    # if GPU
    # parallel_model = multi_gpu_model(model, gpus=2)
    # parallel_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # result = parallel_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
    #                             callbacks=[TensorBoard(log_dir='./../result/log/'+n+'_log')])

    model.save('./../result/'+n+'_model.h5')
    plot_model(model, show_shapes = True, to_file='./../result/'+n+'_model.png')

    return result


def result_curve(result,n):
    # 绘制出结果
    plt.figure
    plt.plot(result.epoch,result.history['acc'],label="acc")
    plt.scatter(result.epoch,result.history['acc'])
    plt.plot(result.epoch,result.history['loss'],label="loss")
    plt.scatter(result.epoch,result.history['loss'],marker='*')
    plt.legend(loc='upper right')
    plt.title(n)
    plt.savefig('./../result/'+n+'_curve.png')
    plt.show()


def Multimodel(cnn_weights_path=None, all_weights_path=None, class_num=119, cnn_no_vary=False):

    input_layer = Input(shape=(200, 200, 3))
    incptionResnet = InceptionResNetV2(include_top=False,weights=None,
                                       input_tensor=input_layer,input_shape=(224,224,3))
    xception = Xception(include_top=False,weights=None,input_tensor=input_layer,input_shape=(224,224,3))

    if cnn_no_vary:
        for i,layer in  enumerate(incptionResnet.layers):
            incptionResnet.layers[i].trainable=False
        for i,layer in enumerate(xception.layers):
            xception.layers[i].trainable=False

    if cnn_weights_path != None:
        incptionResnet.load_weights(cnn_weights_path[0])
        xception.load_weights(cnn_weights_path[1])

    print(incptionResnet.output.shape, xception.output.shape)
    model1 = GlobalMaxPool2D(data_format='channels_last')(incptionResnet.output)
    model2 = GlobalMaxPool2D(data_format='channels_last')(xception.output)

    print(model1.shape, model2.shape)
    # 把top1_model和top2_model连接起来
    x = keras.layers.Concatenate(axis=1)([model1, model2])
    # x = keras.layers.Add()([model1, model2])

    # 全连接层
    x = Dense(units=256*3, activation="relu")(x)
    x = Dense(units=256, activation="relu")(x)
    # x = Dropout(0.5)(x)
    x = Dense(units=class_num, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=x)

    # 加载全部的参数
    if all_weights_path:
        model.load_weights(all_weights_path)

    return model

def main(epochs,batch_size):
    model = Multimodel(cnn_weights_path=None, all_weights_path=None, class_num=119, cnn_no_vary=False)
    n = 'InceptionResNetV2_Xception'
    result = train_model(model,n, epochs, batch_size)
    result_curve(result,n)


if __name__ =="__main__":
    main(epochs=60, batch_size=32)


