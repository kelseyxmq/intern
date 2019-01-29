#-*- coding:utf-8 -*-

from keras.utils import plot_model
from keras.applications import ResNet50
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard
from keras import optimizers
import os
from keras.utils.np_utils import to_categorical
from tfrecords_to_dataset import get_data

'消除警告'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def train_model(model,n,epochs,batch_size):

    x_train,y_train = get_data('./../train.tfrecords')
    y_train = to_categorical(y_train, num_classes=119)

    model.summary()
    rmsprop = optimizers.RMSprop(lr=0.001)

    # if GPU
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
    result = parallel_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                callbacks=[TensorBoard(log_dir='./../result/log/'+n+'_log')])

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


def main(epochs,batch_size):
    model = ResNet50(input_shape=(200,200,3),weights=None,classes=119)
    n = 'ResNet50'
    result = train_model(model,n, epochs, batch_size)
    result_curve(result,n)


if __name__ =="__main__":
    main(epochs=50, batch_size=64)


