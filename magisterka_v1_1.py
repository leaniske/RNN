import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('measurements_join.csv', sep = ';', decimal = ',')
data.drop(['Number of Records', 'ModuleCode', 'MeasurementInterval', 'ModuleName'], inplace = True, axis = 1)
data = data.loc[data.Unit == '°C']
data['MeasurementDateTime'] = pd.to_datetime(data['MeasurementDateTime'], dayfirst = True)
data.sort_values('MeasurementDateTime')
print(data.info())
print(data.head())
data.plot(x = 'MeasurementDateTime', y = 'Value')
plt.show()
# kodowanie wartości tekstowych w kolumnie "Unit"
unit_codes = {unit: i for i, unit in enumerate(data['Unit'].unique())}
data['UnitCode'] = data['Unit'].map(unit_codes)
# podziel dane na zestawy treningowe i testowe
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
# przeskalowanie wartości numerycznych
# przeskalowanie wartości numerycznych
scaler = MinMaxScaler()
train_data.loc[:, 'Value'] = scaler.fit_transform(train_data.loc[:, ['Value']])
test_data.loc[:, 'Value'] = scaler.transform(test_data.loc[:, ['Value']])
# definicja modelu sieci neuronowej
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(24, 2)),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(1)
])
# kompilacja modelu sieci neuronowej
model.compile(optimizer='adam', loss='mse')
# funkcja do przetwarzania danych wejściowych
def prepare_input_data(df, look_back=24):
    X, Y = [], []
    for i in range(look_back, len(df)):
        if df.iloc[i]['SensorId'] != df.iloc[i-look_back]['SensorId']:
            continue
        values = df.iloc[i-look_back:i][['UnitCode', 'Value']].values
        X.append(values)
        Y.append(df.iloc[i]['Value'])
    return np.array(X), np.array(Y)
# przygotowanie danych treningowych i testowych
train_X, train_Y = prepare_input_data(train_data)
test_X, test_Y = prepare_input_data(test_data)
# trenowanie modelu sieci neuronowej
model.fit(train_X, train_Y, epochs=10, batch_size=64)
# ocena modelu sieci neuronowej
test_loss = model.evaluate(test_X, test_Y)
print(f'Test loss: {test_loss:.4f}')
# przewidywanie zmiany temperatury
predictions = model.predict(test_X)
# odwrócenie skalowania wartości numerycznych
predictions = scaler.inverse_transform(predictions)
# wykrycie przypadków zaburzeń przewozu
threshold = 1.0
anomalies = np.where(np.abs(predictions - test_Y.reshape(-1, 1)) > threshold)[0]
print(f'Liczba przypadków zaburzeń przewozu: {len(anomalies)}')