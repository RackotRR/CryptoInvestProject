import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import csv
import seaborn as sbn
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.layers import *
from keras import regularizers
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def create_Xt_Yt(X, y, percentage=0.9):
    p = int(len(X) * percentage)
    X_train = X[0:p]
    Y_train = y[0:p]

    X_train, Y_train = shuffle_in_unison(X_train, Y_train)

    X_test = X[p:]
    Y_test = y[p:]

    return X_train, X_test, Y_train, Y_test

with open("AAPL_clear.csv", encoding='utf-8') as r_file:

    data = pd.read_csv(r_file, delimiter=",")
    #close_price = np.zeros((1134, 1))
    close_price = data[["Adj Close"]].to_numpy()
    data = data.drop(columns=['Date'], axis=1)
    df = pd.DataFrame(data)
    #print(data.values)
    # for index, row in data.iterrows():
    #     close_price[index] = row[5]

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.set_xlim(1, 500)
# ax1.set_ylim(100, 200)
# ax1.plot(close_price)
# ax1.set_xlabel('days')
# ax1.set_ylabel('dollars')
# ax1.set_title('cost')
# plt.show()

# print(df.head())
# print(df.describe())
# print(df.isnull().sum())
# print(df.info())
#
# sbn.pairplot(df, hue='Volume')
# plt.show()
print(close_price)
WINDOW = 30
EMB_SIZE = 1
STEP = 1
FORECAST = 5
X, Y = [], []

for i in range(0, len(close_price), STEP):
    try:
        x_i = close_price[i:i+WINDOW]
        y_i = close_price[i+WINDOW+FORECAST]

        last_close = x_i[WINDOW-1]
        next_close = y_i

        if last_close < next_close:
            y_i = [1, 0]
        else:
            y_i = [0, 1]

    except Exception as e:
        print(e)
        break

    X.append(x_i)
    Y.append(y_i)

X = [(np.array(x) - np.mean(x)) / np.std(x) for x in X]  # comment it to remove normalization
X, Y = np.array(X), np.array(Y)
X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y)

model = Sequential()
model.add(Dense(32, input_dim=30, activity_regularizer=regularizers.l2(0.01)))
#model.add(BatchNormalization())
model.add(ReLU())

#model.add(Dropout(0.5))
model.add(Dense(16, activity_regularizer=regularizers.l2(0.01)))
#model.add(BatchNormalization())
model.add(ReLU())
model.add(Dense(2))
model.add(Activation('softmax'))

opt = Nadam(lr=0.001)

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=25, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=1, save_best_only=True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train,
          epochs = 100,
          batch_size = 128,
          verbose=1,
          validation_data=(X_test, Y_test),
          callbacks=[reduce_lr, checkpointer],
          shuffle=True)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

pred = model.predict(np.array(X_test))
#C = confusion_matrix([np.argmax(y) for y in Y_test], [np.argmax(y) for y in pred])

#print (C / C.astype(np.float).sum(axis=1))

FROM = 0
TO = FROM + 500

original = Y_test[FROM:TO]
predicted = pred[FROM:TO]

plt.plot(original, color='black', label = 'Original data')
plt.plot(predicted, color='blue', label = 'Predicted data')
plt.show()
# plt.plot(rr, color='blue', label = 'Predicted data')
# plt.plot(qw, color='red', label = 'Predicted data')
# plt.legend(loc='best')
# plt.title('Actual and predicted from point %d to point %d of test set' % (FROM, TO))
# plt.show()

