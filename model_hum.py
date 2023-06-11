import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import RMSprop
from keras.callbacks import Callback

df = pd.read_csv('measurements_join_eda.csv', sep=';', decimal=',')

df.drop(['Number of Records', 'ModuleCode', 'MeasurementInterval', 'ModuleName'], inplace=True, axis=1)

df['MeasurementDateTime'] = pd.to_datetime(df['MeasurementDateTime'], dayfirst=True)
df = df.sort_values(by='MeasurementDateTime')
df = df.loc[df.Unit == '%']
df_group = df.groupby(['SensorId']).mean()
print(df_group)

df = df[~df['SensorId'].isin(['602267e77f522d0007fb01cc', '602267e77f522d0007fb01d3', '602267e87f522d0007fb01e2', '602267e87f522d0007fb01e3', '602267e87f522d0007fb01e7'])]
df = df.reset_index()
df.drop('index', axis=1, inplace=True)

df = df[df['MeasurementDateTime'] < '2020-07-01']

Tp = 4750
plt.figure(figsize=(15, 4))
plt.title("Humidity of first {} data points".format(Tp), fontsize=16)
plt.scatter(df['MeasurementDateTime'][:Tp], df['Value'][:Tp], c = df['Value'][:Tp], cmap = 'plasma', s = 2, lw=1)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

plt.figure(figsize=(15, 4))
plt.title("Humidity: all data points", fontsize=16)
plt.scatter(df['MeasurementDateTime'], df['Value'], c = df['Value'], cmap = 'plasma', s = 2, lw=1)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

train = np.array(df['Value'][:Tp])
test = np.array(df['Value'][Tp:])

train = train.reshape(-1, 1)
test = test.reshape(-1, 1)

step = 8


def convertToMatrix(data, step):
    X, Y = [], []
    for i in range(len(data)-step):
        d = i+step
        X.append(data[i:d,])
        Y.append(data[d,])
    return np.array(X), np.array(Y)


def build_simple_rnn(num_units=128, embedding=4, num_dense=32, lr=0.0005):
    model = Sequential()
    model.add(SimpleRNN(units=num_units, input_shape=(1, embedding), activation="relu"))
    model.add(Dense(num_dense, activation="relu"))
    model.add(Dense(num_dense, activation="relu"))
    model.add(Dense(num_dense, activation="relu"))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=lr), metrics=['mse'])
    return model


class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % 50 == 0 and epoch > 0:
            print("Epoch number {} done".format(epoch+1))


# add step elements into train and test
test = np.append(test, np.repeat(test[-1,], step))
train = np.append(train, np.repeat(train[-1,], step))

trainX, trainY = convertToMatrix(train, step)
testX, testY = convertToMatrix(test, step)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model_temp = build_simple_rnn(num_units=128, num_dense=32, embedding=8, lr=0.001)

batch_size = 64
num_epochs = 2000

model_temp.fit(trainX, trainY, epochs=num_epochs, batch_size=batch_size, callbacks=[MyCallback()], verbose=0)

plt.figure(figsize=(7, 5))
plt.title("RMSE loss over epochs", fontsize=16)
plt.plot(np.sqrt(model_temp.history.history['loss']), c='k', lw=2)
plt.grid(True)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Root-mean-squared error", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

trainPredict = model_temp.predict(trainX)
testPredict = model_temp.predict(testX)
predicted = np.concatenate((trainPredict, testPredict), axis=0)
index = df.index.values

plt.figure(figsize=(15, 5))
plt.title("Humidity: Ground truth and prediction together", fontsize=18)
plt.plot(df['MeasurementDateTime'], df['Value'], c='blue')
plt.plot(df['MeasurementDateTime'], predicted, c='orange', alpha=0.75)
plt.legend(['True data', 'Predicted'], fontsize=15)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
