# import lib
import sys
import datetime as dt
import math
from pandas_datareader import data as web
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('fivethirtyeight')

# g et the stock ticker
# TODO read argument
ticker = sys.argv[1]

# get the stock quote 2012-2019
start = dt.datetime(2001, 1, 1)
today = dt.date.today()
df = web.DataReader(ticker, 'yahoo', start, today - dt.timedelta(days=1))
print(df)

# get the number of rows and columns in the data set
df.shape

# visualize the closing price history
plt.figure(figsize=(16, 8))
plt.title('Close price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close price USD ($)', fontsize=18)
plt.show()

# create a new dataframe with only the close column
data = df.filter(['Close'])

# convert the dataframe to a numpy array
dataset = data.values

# the number of rows to train the model on, 80% of it
training_data_len = math.ceil(len(dataset) * 0.8)
print(training_data_len)

# scale the data into vlaue betweee 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)

# create the training data set
# create the scaled training dataset
train_data = scaled_data[0:training_data_len, :]

# split the data into x_train and y_train data sets
x_train = []  # independent var , past 60 values
y_train = []  # dependent var

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 61:
        print(x_train)
        print(y_train)
        print()

# convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# reshape the data into a 3d
print(x_train.shape)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

# build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))  # 25 neuron
model.add(Dense(1))  # 1 neuron

# compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# create the testing data set
# create a new array containing scaled values from index 1968 to ..
# ir descaled dataset
test_data = scaled_data[training_data_len - 60:, :]
# create the data set x_test, y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# convert the data to a numpy array so that it can be used in the model
x_test = np.array(x_test)

# reshape the data to a 3d (for LSTM model)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# get the model predicxted x values
predictions = model.predict(x_test)
# unscaling the values, we want predictions to contrain the same value as y_test set
predictions = scaler.inverse_transform(predictions)

# get the root mean squared error RMSE
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
#rmse = np.sqrt(np.mean(np.power((np.array(y_test)-np.array(predictions)),2)))
#rmse = np.sqrt(((predictions - y_test) ** 2).mean())
print("rmse={}".format(rmse))

# plot data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# visualize the validation data
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Data', fontsize=18)
plt.ylabel('Close Price USD ($)')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Trained', 'Actual', 'Predictions'], loc='lower right')
plt.show()

# show the validation price and the predicted price, actual vs predicted
print(valid)

# predict the price for next dates
quote = web.DataReader(ticker, 'yahoo', start, today)
# create a new dataframe
new_df = quote.filter(['Close'])
# get the last 60 day closing value and convert the dataframe to an array
last_60_days = new_df[-60:].values
# Scale the data to value between 0 and 1 for the model
last_60_days_scaled = scaler.transform(last_60_days)
# create an empty list
x_test = []
# append the past 60 days to x_test list (the scaled ones)
x_test.append(last_60_days_scaled)
# convert the x_test dataset to a numpy array
x_test = np.array(x_test)
# reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# get the predicted scaled price
pred_price = model.predict(x_test)
# undo the scaling
pred_price = scaler.inverse_transform(pred_price)
# print the predicted price
print("predicted price is {}".format(pred_price))
