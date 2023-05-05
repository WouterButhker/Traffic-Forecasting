import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, utils
import matplotlib.pyplot as plt
def splitSequence(seq, n_steps):

    #Declare X and y as empty list
    X = []
    y = []

    for i in range(len(seq)):
        #get the last index
        lastIndex = i + n_steps

        #if lastIndex is greater than length of sequence then break
        if lastIndex > len(seq) - 1:
            break

        #Create input and output sequence
        seq_X, seq_y = seq[i:lastIndex], seq[lastIndex]

        #append seq_X, seq_y in X and y list
        X.append(seq_X)
        y.append(seq_y)
        pass    #Convert X and y into numpy array
    X = np.array(X)
    y = np.array(y)

    return X,y
    pass



def main():
    df = pd.read_csv("data/K120 tellen 2019-11-01 tot 2019-11-30.csv", sep=";", parse_dates=["K120"]).dropna()
    columns = df.columns.values
    val_columns = columns[1:]
    df = df.rename(columns={columns[0]: "datetime"})
    df[val_columns] = df[val_columns].applymap(np.int64)



    # df = df.groupby(df["datetime"].dt.hour).sum()
    df = df.transpose()
    print(df.shape)
    df = df[1:]
    print(df.shape)
    arr = df.transpose().to_numpy()
    #arr = arr.flatten()
    # print(df.info())
    print(arr)
    train, test = np.split(arr, 2)

    train = np.asarray(train).astype('float32')
    test = np.asarray(test).astype('float32')

    # df.plot(kind="bar", y=["021", "022", "023"], ylabel="sum of vehicles", xlabel="hour of day")
    # plt.show()

    steps = 25
    x_train, y_train = splitSequence(train, n_steps=steps)
    x_test, y_test = splitSequence(test, steps)

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 10
    ###Define model###
    inp = layers.Input((steps, n_features))
    epochs = 50

    # LSTMs
    x = layers.LSTM(50)(inp)
    out = layers.Dense(n_features)(x)

    model = Model(inp, out)


    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, verbose=1)

    results = model.evaluate(x_test, y_test, batch_size=128)

    print("test loss, test acc:", results)
    predicted_y = model.predict(x_test)

    for i in range(10):
        plt.title(f"Sensor: {i}, epochs: {epochs}")
        plt.plot(y_test[:150, i], 'C2')
        plt.plot(predicted_y[:150, i], 'C20')
        plt.show()


if __name__ == '__main__':
    main()


