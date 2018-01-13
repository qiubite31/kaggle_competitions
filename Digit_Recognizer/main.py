from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# label,pixel0,pixel1,pixel2,pixel3
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# training
'''
X = train_df.drop(['label'], axis=1)
Y = train_df[['label']]

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.33, random_state=42)

model = Sequential()

x_train = np.reshape(x_train, (x_train.shape[0], -1))/255
x_val = np.reshape(x_val, (x_test.shape[0], -1))/255

y_train = np.eye(10)[y_train['label'].values.reshape(-1)]
y_val = np.eye(10)[y_val['label'].values.reshape(-1)]

x_train = x_train.as_matrix()
x_val = x_val.as_matrix()

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

model.add(Dense(units=256, activation='relu', input_dim=28*28))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',  # mean_squared_error
              optimizer='Adam',  # Adam
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32)

print('training finish!')

# evaluation
loss_and_metrics = model.evaluate(x_train, y_train, batch_size=128)
print(loss_and_metrics)

loss_and_metrics = model.evaluate(x_val, y_val, batch_size=128)
print(loss_and_metrics)
'''

# use all train set to training model
x_train_all = train_df.drop(['label'], axis=1)
y_train_all = train_df[['label']]

model = Sequential()

x_train_all = np.reshape(x_train_all, (x_train_all.shape[0], -1))/255
y_train_all = np.eye(10)[y_train_all['label'].values.reshape(-1)]

x_train_all = x_train_all.as_matrix()

model.add(Dense(units=256, activation='relu', input_dim=28*28))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

model.fit(x_train_all, y_train_all, epochs=10, batch_size=32)


# predict test data set
x_test_all = test_df
x_test_all = np.reshape(x_test_all, (x_test_all.shape[0], -1))/255
x_test_all = x_test_all.as_matrix()

predicts = model.predict(x_test_all)
predict_df = pd.DataFrame(data=predicts).idxmax(axis=1)
predict_df.to_csv('result.csv')
