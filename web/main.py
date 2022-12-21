from pylab import *
import pandas as pd
import tensorflow as tf
from datetime import date, timedelta


##############################################################################################################
class Neuron:
    will_it_grow = True
    result: float
    dataset_new: []
    dots = ""

    def univariate_data(self, dataset, start_index, end_index, history_size, target_size):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i - history_size, i)
            data.append(np.reshape(dataset[indices], (history_size, 1)))
            labels.append(dataset[i + target_size])
        return np.array(data), np.array(labels)

    def __init__(self, dataset_name):

        dataset = pd.read_csv("web/" + dataset_name, delimiter=",")
        dataset.set_index('Date', inplace=True)
        dataset.index = pd.to_datetime(dataset.index)
        dataset.drop(['Open', 'Close', 'Low', 'High', 'Volume'], axis=1, inplace=True)
        dataset.plot(subplots=True)
        # plt.show()

        self.dataset_new = dataset.values[-30:]

        a = 29
        for it in self.dataset_new:
            date_30 = date.today() - timedelta(days=a)
            a = a - 1
            self.dots = self.dots + "['" + str(date_30.day) + "." + str(date_30.month) + "', " + str(round(float(it), 2)) + "],"

        train = dataset[:1000]
        test = dataset[1000:]

        dataset = (dataset - train.min()) / (train.max() - train.min())

        dataset = dataset.values
        past_history = 15
        future_target = 0
        x_train, y_train = self.univariate_data(dataset, 0, 1000, past_history, future_target)
        x_test, y_test = self.univariate_data(dataset, 1000, 1134, past_history, future_target)

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

        for i in range(len(x[0].numpy())):
            Xx = np.append(Xx, (x[0][i] * (train.max() - train.min()) + train.min()).numpy())

        self.result = float(simple_lstm_model.predict(x)[len(y) - 1] * (train.max() - train.min()) + train.min())

        if self.result > self.dataset_new[-1:]:
            self.will_it_grow = True
        else:
            self.will_it_grow = False

#######################################################################################################################
