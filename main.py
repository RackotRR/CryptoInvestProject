# import matplotlib.pylab as plt
# import numpy as np
# import pandas as pd
# import csv
# import seaborn as sbn
# from keras.models import Sequential
# from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
# from keras.optimizers import RMSprop, Adam, SGD, Nadam
# from keras.layers import *
# from keras import regularizers
# from keras.utils import normalize
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn import preprocessing
import csv
import numpy as np
import matplotlib
from matplotlib.ticker import MultipleLocator
from pylab import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from sklearn import preprocessing
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


dataset = pd.read_csv("AAPL_clear.csv", delimiter=",")
dataset.set_index('Date', inplace=True)
dataset.index = pd.to_datetime(dataset.index)
dataset.drop(['Open', 'Close', 'Low', 'High', 'Volume'], axis=1, inplace=True)
dataset.plot(subplots=True)
plt.show()

train = dataset[:1000]
test = dataset[1000:]
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
x_train, y_train = univariate_data(dataset, 0, 1000, past_history, future_target)
x_test, y_test = univariate_data(dataset, 1000, 1134, past_history, future_target)

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

EVALUATION_INTERVAL = 100
EPOCHS = 35

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
        file_writer.writerow([float((x[i][14] * (train.max() - train.min()) + train.min()).numpy()),
                              float(y[i].numpy() * (train.max() - train.min()) + train.min()),
                              float(simple_lstm_model.predict(x)[i] * (train.max() - train.min()) + train.min()),
                              Yx,
                              Zx])
# show_plot([x_train[0], y_train[0],baseline(x_train[0])], 0, 'Sample Example')
# plt.show()

# def shuffle_in_unison(a, b):
#     assert len(a) == len(b)
#     shuffled_a = np.empty(a.shape, dtype=a.dtype)
#     shuffled_b = np.empty(b.shape, dtype=b.dtype)
#     permutation = np.random.permutation(len(a))
#     for old_index, new_index in enumerate(permutation):
#         shuffled_a[new_index] = a[old_index]
#         shuffled_b[new_index] = b[old_index]
#     return shuffled_a, shuffled_b
#
# def create_Xt_Yt(X, y, percentage=0.9):
#     p = int(len(X) * percentage)
#     X_train = X[0:p]
#     Y_train = y[0:p]
#
#     X_train, Y_train = shuffle_in_unison(X_train, Y_train)
#
#     X_test = X[p:]
#     Y_test = y[p:]
#
#     return X_train, X_test, Y_train, Y_test
#
# with open("AAPL_clear.csv", encoding='utf-8') as r_file:
#     data = pd.read_csv(r_file, delimiter=",")
#     # three_sigma_vol = 3 * data["Volume"].std()
#     # three_sigma_cls = 3 * data["Adj Close"].std()
#     # i=0
#     # for index, row in data.iterrows():
#     #     vol = row["Volume"]
#     #     cls = row["Adj Close"]
#     #
#     #     if vol < data["Volume"].mean() - three_sigma_vol or \
#     #             vol > data["Volume"].mean() + three_sigma_vol or \
#     #             cls < data["Adj Close"].mean() - three_sigma_cls or \
#     #             cls > data["Adj Close"].mean() + three_sigma_cls:
#     #         i+=1
#     # print(i)
#     # exit()
#     close_price = data[["Adj Close"]].to_numpy()
#     data = data.drop(columns=['Date'], axis=1)
#     volume = data[["Volume"]].to_numpy()
#
# # fig = plt.figure()
# # ax1 = fig.add_subplot(111)
# # ax1.set_xlim(1, 500)
# # ax1.set_ylim(100, 200)
# # ax1.plot(close_price)
# # ax1.set_xlabel('days')
# # ax1.set_ylabel('dollars')
# # ax1.set_title('cost')
# # plt.show()
#
# # print(df.head())
# # print(df.describe())
# # print(df.isnull().sum())
# # print(df.info())
# #
#
# # sbn.pairplot(data, hue='Volume')
# # plt.show()
# #
# # sbn.boxplot(volume)
# # plt.show()
# # sbn.boxplot(close_price)
# # plt.show()
#
# #print(close_price)
#
# WINDOW = 30
# EMB_SIZE = 1
# STEP = 1
# FORECAST = 5
# X, Y = [], []
#
# for i in range(0, len(close_price), STEP):
#     try:
#         x_i = close_price[i:i+WINDOW]
#         y_i = close_price[i+WINDOW+FORECAST]
#
#         last_close = x_i[WINDOW-1]
#         next_close = y_i
#
#         if last_close < next_close:
#             y_i = [1, 0]
#         else:
#             y_i = [0, 1]
#
#     except Exception as e:
#         print(e)
#         break
#
#     X.append(x_i)
#     Y.append(y_i)
# X = np.resize(X, (30, 30))
# Y = np.resize(Y, (30, 2))
# transformer = preprocessing.Normalizer().fit(X)
# #X = [(np.array(x) - np.mean(x)) / np.std(x) for x in X]  # comment it to remove normalization
# X = transformer.transform(X)
# print(X)
# X, Y = np.array(X), np.array(Y)
# X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y)
#
# model = Sequential()
# model.add(Dense(32, input_dim=30, activity_regularizer=regularizers.l2(0.01)))
# model.add(BatchNormalization())
# model.add(ReLU())
# model.add(Dropout(0.5))
# model.add(Dense(16, activity_regularizer=regularizers.l2(0.01)))
# model.add(BatchNormalization())
# model.add(ReLU())
# model.add(Dense(2))
# model.add(Activation('linear'))
#
# opt = Adam()
#
# reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=25, min_lr=0.000001, verbose=1)
# #checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=0, save_best_only=True)
# model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
#
# history = model.fit(X_train, Y_train,
#           epochs = 100,
#           batch_size = 128,
#           verbose=0,
#           validation_data=(X_test, Y_test),
#           callbacks=[reduce_lr],
#           shuffle=True)
#
# plt.figure()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='best')
# plt.show()
#
# plt.figure()
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='best')
# plt.show()
#
# # pred = model.predict(np.array(X_test))
# # #C = confusion_matrix([np.argmax(y) for y in Y_test], [np.argmax(y) for y in pred])
# #
# # #print (C / C.astype(np.float).sum(axis=1))
# #
# # FROM = 0
# # TO = FROM + 500
# #
# # original = Y_test[FROM:TO]
# # predicted = pred[FROM:TO]
# #
# # plt.plot(original, color='black', label = 'Original data')
# # plt.plot(predicted, color='blue', label = 'Predicted data')
# # plt.show()
# # plt.plot(rr, color='blue', label = 'Predicted data')
# # plt.plot(qw, color='red', label = 'Predicted data')
# # plt.legend(loc='best')
# # plt.title('Actual and predicted from point %d to point %d of test set' % (FROM, TO))
# # plt.show()
