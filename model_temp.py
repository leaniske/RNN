import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import RMSprop
from keras.callbacks import Callback
from sklearn.metrics import precision_score, recall_score, f1_score

# Wczytanie danych z pliku CSV.
df = pd.read_csv('measurements_join_eda.csv', sep=';', decimal=',')

# Usunięcie niepotrzebnych kolumn.
df.drop(['Number of Records', 'ModuleCode', 'MeasurementInterval', 'ModuleName'], inplace=True, axis=1)

# Zmiana typu danych na datetime.
df['MeasurementDateTime'] = pd.to_datetime(df['MeasurementDateTime'], dayfirst=True)
# Sortowanie danych względem daty.
df = df.sort_values(by='MeasurementDateTime')
# Wybranie zapisów, w których jednostką jest stopieć Celsjusza.
df = df.loc[df.Unit == '°C']
# Średnie wartości dla poszczególnych czujników.
df_group = df.groupby(['SensorId']).mean()
print(df_group)

# Wykluczenie czujników, których uwzględnienie zaburzyłoby wyniki predykcji.
df = df[~df['SensorId'].isin(['602267e77f522d0007fb01cc', '602267e77f522d0007fb01d2', '602267e77f522d0007fb01d3', '602267e87f522d0007fb01d6', '602267e87f522d0007fb01da', '602267e87f522d0007fb01db', '602267e87f522d0007fb01e2', '602267e87f522d0007fb01e3', '602267e87f522d0007fb01e7'])]
# Ustalenie indexu na nowo.
df = df.reset_index()
df.drop('index', axis=1, inplace=True)

# Wybranie rekordów pierwszego okresu użytkowania.
df = df[df['MeasurementDateTime'] < '2020-07-01']

# Określenie liczby punktów treningowych jako Tp.
Tp = 4750
# Wykres treningowych punktów pomiarowych.
plt.figure(figsize=(15, 4))
plt.title("Temperature of first {} data points".format(Tp), fontsize=16)
plt.scatter(df['MeasurementDateTime'][:Tp], df['Value'][:Tp], c = df['Value'][:Tp], cmap = 'plasma', s = 2, lw=1)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Wykres wszystkich punktów pomiarowych.
plt.figure(figsize=(15, 4))
plt.title("Temperature all data points", fontsize=16)
plt.scatter(df['MeasurementDateTime'], df['Value'], c = df['Value'], cmap = 'plasma', s = 2, lw=1)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Przygotowanie danych treningowych i testowych.
train = np.array(df['Value'][:Tp])
test = np.array(df['Value'][Tp:])

train = train.reshape(-1, 1)
test = test.reshape(-1, 1)

step = 8


# Funkcja pomocnicza do konwersji danych na macierze.
def convertToMatrix(data, step):
    X, Y = [], []
    for i in range(len(data)-step):
        d = i+step
        X.append(data[i:d,])
        Y.append(data[d,])
    return np.array(X), np.array(Y)


# Funkcja do budowy modelu RNN.
def build_simple_rnn(num_units=128, embedding=4, num_dense=32, lr=0.001):
    model = Sequential()
    model.add(SimpleRNN(units=num_units, input_shape=(1, embedding), activation="relu"))
    model.add(Dense(num_dense, activation="relu"))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=lr), metrics=['mse'])
    return model


# Klasa, która informuje o zakończeniu danych ecykli obliczeń.
class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % 50 == 0 and epoch > 0:
            print("Epoch number {} done".format(epoch+1))


# Dodanie elementów o długości 'step' do danych treningowych i testowych.
test = np.append(test, np.repeat(test[-1,], step))
train = np.append(train, np.repeat(train[-1,], step))

# Konwersja danych treningowych i testowych na macierze.
trainX, trainY = convertToMatrix(train, step)
testX, testY = convertToMatrix(test, step)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Inicjalizacja modelu sieci neuronowej.
model_temp = build_simple_rnn(num_units=128, num_dense=32, embedding=8, lr=0.001)

batch_size = 50
num_epochs = 2000

# Trenowanie modelu sieci neuronowej.
model_temp.fit(trainX, trainY, epochs=num_epochs, batch_size=batch_size, callbacks=[MyCallback()], verbose=0)

# Wykres wartości RMSE w zależności od epoch.
plt.figure(figsize=(7, 5))
plt.title("RMSE loss over epochs", fontsize=16)
plt.plot(np.sqrt(model_temp.history.history['loss']), c='k', lw=2)
plt.grid(True)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Root-mean-squared error", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Predykcje na danych treningowych i testowych.
trainPredict = model_temp.predict(trainX)
testPredict = model_temp.predict(testX)
predicted = np.concatenate((trainPredict, testPredict), axis=0)
index = df.index.values

# Porównanie rzeczywistych wartości do predykcji.
plt.figure(figsize=(15, 5))
plt.title("Temperature: Ground truth and prediction together", fontsize=18)
plt.plot(df['MeasurementDateTime'], df['Value'], c='blue')
plt.plot(df['MeasurementDateTime'], predicted, c='orange', alpha=0.75)
plt.legend(['True data', 'Predicted'], fontsize=15)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
