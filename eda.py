import pandas as pd
import numpy as np


df = pd.read_csv('measurements_join_eda.csv', sep = ';', decimal = ',')
df.drop(['Number of Records', 'ModuleCode', 'MeasurementInterval', 'ModuleName'], inplace = True, axis = 1)
df = df.loc[df.Unit == '°C']
df['MeasurementDateTime'] = pd.to_datetime(df['MeasurementDateTime'], dayfirst = True)
df = df.sort_values(by='MeasurementDateTime')
df = df.reset_index()
# report = ProfileReport(df)
# report.to_file("report.html")
# print(df['SensorId'].unique())
df_group = df.groupby(['SensorId']).mean()
print(df_group)
# for item in df['SensorId'].unique():




# Tp = 20000
# plt.figure(figsize=(15,4))
# plt.title("Temperature of first {} data points".format(Tp),fontsize=16)
# plt.plot(df['Value'][:Tp],c='k',lw=1)
# plt.grid(True)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.show()