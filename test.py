#-*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np
from keras.models import load_model
import csv
import codecs

'消除警告'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def get_img_data(img_path,size=200):

    im = Image.open(img_path)
    im = im.resize((size,size),Image.ANTIALIAS)
    # im = im.convert("L")
    im = np.array(im).astype('float32')
    img_data = im.reshape(1,size,size,3)
    return img_data


def predict_img(im0,im1,model):

    print("="*65)
    y0 = model.predict(im0)
    y0 = np.argmax(y0,axis=1)
    y1 = model.predict(im1)
    y1 = np.argmax(y1,axis=1)

    if y0==y1:
        res='yes'
    else :
        res='no'

    print(y0,y1,res)

    return res

def main(net):

    pt = './../test_data/'
    model = load_model('./../result/'+net+'model.h5')
    x = []
    for i in range(119):
        im0 = get_img_data(pt+str(i)+'/0.jpg')
        im1 = get_img_data(pt+str(i)+'/1.jpg')
        result = predict_img(im0, im1, model)
        n = {'a':i,'b':result}
        x.append(n)

    fileName = './../result/'+net+'test_result.csv'
    with codecs.open(fileName, 'w', 'utf-8') as csvfile:
        filednames = ['测试样本', '测试结果']
        writer = csv.DictWriter(csvfile, fieldnames=filednames)

        writer.writeheader()
        for n in x:
            writer.writerow({'测试样本': n['a'], '测试结果': n['b']})


if __name__ =="__main__":
    net = "InceptionV3_"
    print("="*35+net+"="*35)
    main(net)

    net = "ResNet50_"
    print("="*35+net+"="*35)
    main(net)
