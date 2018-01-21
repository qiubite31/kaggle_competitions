import glob
import os
from PIL import Image, ImageOps
import re
import numpy as np
import pandas as pd
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model

img_list = glob.glob('train/*/*.png')
labels = []
imgs = []
for img_name in img_list:
    labels.append(img_name.split('\\')[1])
    img_temp = Image.open(img_name)
    img_new = ImageOps.fit(img_temp, (96, 96), Image.ANTIALIAS).convert('RGB')
    imgs.append(img_new)


img_raw = np.array([np.array(im) for im in imgs])
img_raw = img_raw.reshape(img_raw.shape[0], 96, 96, 3) / 255
lb = LabelBinarizer().fit(labels)
label = lb.transform(labels)

x_train, x_val, y_train, y_val = train_test_split(img_raw, label, test_size=0.33, random_state=42)

# x_train = img_raw
# y_train = label

from keras.layers import Dropout, Input, Dense, Activation, GlobalMaxPooling2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam

input = Input((96, 96, 3))
layer = Conv2D(32, (3, 3))(input)
layer = BatchNormalization(axis=3)(layer)
layer = Activation('relu')(layer)
layer = Conv2D(32, (3, 3))(layer)
layer = BatchNormalization(axis=3)(layer)
layer = Activation('relu')(layer)
layer = MaxPooling2D((2, 2), strides=(2, 2))(layer)
layer = Conv2D(64, (3, 3))(layer)
layer = BatchNormalization(axis = 3)(layer)
layer = Activation('relu')(layer)
layer = Conv2D(64, (3, 3))(layer)
layer = BatchNormalization(axis = 3)(layer)
layer = Activation('relu')(layer)
layer = GlobalMaxPooling2D()(layer)

layer = Dense(128, activation='relu')(layer)
layer = Dense(64, activation='relu')(layer)
layer = Dense(32, activation='relu')(layer)
layer = Dense(12, activation='softmax')(layer)

model = Model(input=input, output=layer)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=15, batch_size=64)

model.save('my_model2.h5')

loss_and_metrics = model.evaluate(x_train, y_train, batch_size=64)
print('test finish')
print(loss_and_metrics)

loss_and_metrics = model.evaluate(x_val, y_val, batch_size=64)
print('validation finish!')
print(loss_and_metrics)

# print('finish!!!!!!!!!!!!')

# using train/val
# train loss: 0.29627643985466562
# test acc:   0.89094908892321634
# val   loss: 0.72460279172780562
# test acc:   0.75765306122448983
'''
model = load_model('my_model.h5')

img_list = glob.glob('test/*.png')
imgs = []
img_names = []
for img_name in img_list:
    img_names.append(img_name.split('\\')[1])
    img_temp = Image.open(img_name)
    img_new = ImageOps.fit(img_temp, (48, 48), Image.ANTIALIAS).convert('RGB')
    imgs.append(img_new)

lb = LabelBinarizer().fit(labels)
img_raw = np.array([np.array(im) for im in imgs])
img_raw = img_raw.reshape(img_raw.shape[0], 48, 48, 3) / 255

x_test = img_raw

predict_result = model.predict(x_test)
predicts = lb.inverse_transform(predict_result)
predict_df = pd.DataFrame(data={'file': img_names, 'species': predicts})
predict_df.to_csv('result.csv', index=False)


print('test finish!')
'''