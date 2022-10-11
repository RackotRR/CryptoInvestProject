import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten

def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
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

with open("AAPL.csv", encoding='utf-8') as r_file:
    data = csv.reader(r_file, delimiter=",")
    count = 0
    close_price = np.zeros((255, 1))
    for row in data:
        if count != 0:
            close_price[count] = row[5]
        count += 1

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlim(1, 250)
ax1.set_ylim(100, 200)
ax1.plot(close_price)
plt.show()

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

#model = Sequential()
#X = [(np.array(x) - np.mean(x)) / np.std(x) for x in X]
#close_price_diffs = close.price.pct_change()p