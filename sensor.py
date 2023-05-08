import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

tf.random.set_seed(7)
df = pd.read_csv("data/K120 tellen 2019-11-01 tot 2019-11-30.csv", sep=";", parse_dates=["K120"])
df.fillna(method='ffill', inplace=True)
columns = df.columns.values
val_columns = columns[1:]
df = df.rename(columns={columns[0]: "datetime"})
df[val_columns] = df[val_columns].applymap(np.int64)

#take the third column
df023 = df.iloc[:,3]
df023_arr = df023.to_numpy()

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(df023_arr.reshape(-1, 1))

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))




# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
 dataX, dataY = [], []
 for i in range(len(dataset)-look_back-1):
  a = dataset[i:(i+look_back), 0]
  dataX.append(a)
  dataY.append(dataset[i + look_back, 0])
 return np.array(dataX), np.array(dataY)



if __name__ == '__main__':
 look_back = 500
 trainX, trainY = create_dataset(train, look_back)
 testX, testY = create_dataset(test, look_back)
 trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
 testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

 # create and fit the LSTM network
 model = Sequential()
 model.add(LSTM(4, input_shape=(1, look_back)))
 model.add(Dense(1))
 model.compile(loss='mean_squared_error', optimizer='adam')
 model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

 # make predictions
 trainPredict = model.predict(trainX)
 testPredict = model.predict(testX)
 # invert predictions
 trainPredict = scaler.inverse_transform(trainPredict)
 trainY = scaler.inverse_transform([trainY])
 testPredict = scaler.inverse_transform(testPredict)
 testY = scaler.inverse_transform([testY])
 # calculate root mean squared error
 trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
 print('Train Score: %.2f RMSE' % (trainScore))
 testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
 print('Test Score: %.2f RMSE' % (testScore))

 # shift train predictions for plotting
 trainPredictPlot = np.empty_like(dataset)
 trainPredictPlot[:, :] = np.nan
 trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
 # shift test predictions for plotting
 testPredictPlot = np.empty_like(dataset)
 testPredictPlot[:, :] = np.nan
 testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
 # plot baseline and predictions
 plt.plot(scaler.inverse_transform(dataset))
 plt.plot(trainPredictPlot)
 plt.plot(testPredictPlot)
 plt.show()
