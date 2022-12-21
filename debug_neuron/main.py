import csv
import numpy as np
import matplotlib
from matplotlib.ticker import MultipleLocator
from pylab import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
# from sklearn import preprocessing
import tensorflow as tf
import os


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)


def create_time_steps(length):
    return list(range(-length, 0))


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel('Time-Step')
    # plt.ylim(min(plot_data[0]) - 0.2, max(plot_data[0]) + 0.2)
    return plt


def baseline(dataset):
    return np.mean(dataset)

df = pd.read_csv('AAPL.csv')

volume = df[['Volume']]
close = df[['Adj Close']]

for index, row in df.iterrows():
      vol = row["Volume"]
      cls = row["Adj Close"]
      if  float(volume.quantile(0.025)) > vol or float(volume.quantile(0.975)) < vol or \
      float(close.quantile(0.025)) > cls or float(close.quantile(0.975)) < cls:
            df = df.drop(labels=[index], axis=0)

df.to_csv("AAPL_clear.csv", index=False)

dataset = pd.read_csv("AAPL_clear.csv", delimiter=",")
dataset.set_index('Date', inplace=True)
dataset.index = pd.to_datetime(dataset.index)
dataset.drop(['Open', 'Close', 'Low', 'High', 'Volume'], axis=1, inplace=True)
dataset.plot(subplots=True)
plt.show()

START = 0
END = dataset['Adj Close'].count()
SEPARATE = int(dataset['Adj Close'].count() * 0.9)

train = dataset[:SEPARATE]
test = dataset[SEPARATE:]
# d = preprocessing.normalize(dataset)
# scaled_df = pd.DataFrame(d, columns = 'Adj Close')
# scaled_df.head()

# Min - Max Normalization
dataset = (dataset - train.min()) / (train.max() - train.min())

# Median Normalization
# dataset = dataset / train.median()

# dataset = (dataset - train.mean())/train.std()

# Z-Score Normalization
# dataset = (dataset - np.average(train)) / np.std(train)

# Sigmoid Normalization
# dataset = 1.0/(1+np.exp(-(dataset)))

dataset = dataset.values
past_history = 15
future_target = 0
x_train, y_train = univariate_data(dataset, START, SEPARATE, past_history, future_target)
x_test, y_test = univariate_data(dataset, SEPARATE, END, past_history, future_target)

BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train.shape[-2:]),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')
for x, y in val_univariate.take(1):
    print(simple_lstm_model.predict(x).shape)

EVALUATION_INTERVAL = 200
EPOCHS = 15

history = simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)
Xx = np.array([])

# print(history.history.keys())
plt.figure()
plt.plot(simple_lstm_model.history.history['loss'])
plt.plot(simple_lstm_model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()


# for i in range(len(y)):
#   print(y[i] + " -- " + simple_lstm_model.predict(x)[i])
# exit()
for i in range(len(x[0].numpy())):
    Xx = np.append(Xx, (x[0][i] * (train.max() - train.min()) + train.min()).numpy())
Yy = y[0].numpy() * (train.max() - train.min()) + train.min()
Zz = simple_lstm_model.predict(x)[0] * (train.max() - train.min()) + train.min()

for x, y in val_univariate.take(1):
    plot = show_plot([Xx, Yy,
                      Zz], 0, 'Simple LSTM model')
    plot.show()
#for i in range(len(y)):
    #print(str(float((x[i][14] * (train.max() - train.min()) + train.min()).numpy())) + "   " + str(float(y[i].numpy() * (train.max() - train.min()) + train.min())))
   # print(float(simple_lstm_model.predict(x)[i] * (train.max() - train.min()) + train.min()))
print(float(abs(Yy - Zz)))
print(float(abs(Yy - Zz) * 100) / float(abs(Yy)))
a = 0
with open("AAPL_predict.csv", mode="w", encoding="utf-8") as w_file:
    file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
    file_writer.writerow(["Значение X", "Значение Y", "Значение Z", "Рост/падение", "Предсказанные рост/падение"])
    for i in range(len(y)):
        if float(y[i].numpy() * (train.max() - train.min()) + train.min()) > float((x[i][14] * (train.max() - train.min()) + train.min()).numpy()):
            Yx = "Рост"
        else:
            Yx = "Падение"
        if float(simple_lstm_model.predict(x)[i] * (train.max() - train.min()) + train.min()) > float((x[i][14] * (train.max() - train.min()) + train.min()).numpy()):
            Zx = "Рост"
        else:
            Zx = "Падение"

        if Yx == Zx:
            a += 1
        file_writer.writerow([float((x[i][14] * (train.max() - train.min()) + train.min()).numpy()),
                              float(y[i].numpy() * (train.max() - train.min()) + train.min()),
                              float(simple_lstm_model.predict(x)[i] * (train.max() - train.min()) + train.min()),
                              Yx,
                              Zx])

