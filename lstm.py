import pandas as pd
import numpy as np
import os
import random
import typing
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.losses import MeanSquaredError
from keras.optimizers import Adam


DATA_PATH = 'data_cars/'
# all_files = os.listdir(DATA_PATH)
all_files = ['K120.csv', 'K140.csv', 'K159.csv', 'K405.csv', 'K406.csv', 'K701.csv', 'K703.csv', 'K709.csv']
print(all_files)

bad_files = [x for x in os.listdir(DATA_PATH) if x not in all_files]
print(bad_files)

all_dataframes = []
for index, file in enumerate(os.listdir(DATA_PATH)):
    print(f"Reading file: {file}")
    file_name = file.split('.')[0]
    df = pd.read_csv(DATA_PATH + file, sep=';')

    df['date'] = pd.to_datetime(df[file_name], format='%Y-%m-%d %H:%M')
    df = df.drop(columns=[file_name])

    df = df.set_index('date')
    df.columns = [f"{file_name}_{col}" for col in df.columns if col != 'date']
    all_dataframes.append(df)
    # print(f"Finished reading file: {file}, shape = {df.shape}")

combined_df = pd.concat(all_dataframes, axis=1)
# combined_df.fillna(method='ffill', inplace=True)
print("na: " ,combined_df.isna().sum().sum())
# combined_df.interpolate(method='linear', inplace=True, limit=3)
# combined_df['hour'] = combined_df.index.hour
# combined_df['day_of_week'] = combined_df.index.dayofweek

# combined_df = combined_df[:]
# print(combined_df)
# print(all_dataframes)
# print(combined_df.isnull().sum().sum())

data = np.array(combined_df, dtype=float)[:, :]
print(data.shape)

sensor_data = np.transpose(data)



all_dataframes = []
for index, file in enumerate(all_files):
    print(f"Reading file: {file}")
    file_name = file.split('.')[0]
    df = pd.read_csv(DATA_PATH + file, sep=';')

    df['date'] = pd.to_datetime(df[file_name], format='%Y-%m-%d %H:%M')
    df = df.drop(columns=[file_name])

    df = df.set_index('date')
    df.columns = [f"{file_name}_{col}" for col in df.columns if col != 'date']
    all_dataframes.append(df)
    # print(f"Finished reading file: {file}, shape = {df.shape}")

combined_df = pd.concat(all_dataframes, axis=1)
# combined_df.fillna(method='ffill', inplace=True)
combined_df.interpolate(method='linear', inplace=True, limit=3)
# combined_df['hour'] = combined_df.index.hour
# combined_df['day_of_week'] = combined_df.index.dayofweek

# combined_df = combined_df[:]
print(combined_df)
# print(all_dataframes)
# print(combined_df.isnull().sum().sum())

data = np.array(combined_df, dtype=float)[:, :]
print(data.shape)
# data = data[:,2]
# scaler = StandardScaler()
# scaler = MinMaxScaler(feature_range=(0, 1))
#Don't transform the time labels -> this way the scaler also works inversely on prediction data because shapes are different otherwise
# data = scaler.fit_transform(data)
# data = scaler.fit_transform(data.reshape(-1, 1))

sensor_data = np.transpose(data)

scalers = []
for i in range(len(sensor_data)):
    scaler = StandardScaler()
    sensor_data[i] = scaler.fit_transform(sensor_data[i].reshape(-1, 1)).reshape(-1)
    scalers.append(scaler)

print(sensor_data.shape)

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

        # Create input and output sequence
        # Last 2 columns are time of day and day of week
        seq_X, seq_y = seq[i:lastIndex], seq[lastIndex]

        #append seq_X, seq_y in X and y list
        X.append(seq_X)
        y.append(seq_y)
        #Convert X and y into numpy array
    X = np.array(X)
    y = np.array(y)

    return X,y

def mergeSequences(X, y):
    # Flatten the first sequence
    mergedSeq = np.empty((len(X), len(X[0][0]) + len(y[0])))

    for i in range(len(X)):
        newArr = np.concatenate((X[i][0], y[i]), axis=0)
        mergedSeq[i] = newArr

    # Convert the list to a numpy array


    return mergedSeq


def shuffle_arrays(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], p

def unshuffle_arrays(a, b, p):
    return a[np.argsort(p)], b[np.argsort(p)]

# num_of_steps = data.shape[0]
print(f"sensor_data shape: {sensor_data.shape}")

train_size = 0.6
val_size = 0.15
shuffle = False
look_back = 80

num_of_steps = sensor_data.shape[1] - look_back

num_train = int(num_of_steps * train_size)
num_val = int(num_of_steps * val_size)
num_test = num_of_steps - num_train - num_val

print(f"num_train: {num_train}")
print(f"num_val: {num_val}")
print(f"num_test: {num_test}")


x_train = np.empty((sensor_data.shape[0], num_train, look_back))
y_train = np.empty((sensor_data.shape[0], num_train))
x_val =  np.empty((sensor_data.shape[0], num_val, look_back))
y_val = np.empty((sensor_data.shape[0], num_val))
x_test = np.empty((sensor_data.shape[0], num_test, look_back))
y_test = np.empty((sensor_data.shape[0], num_test))
permutations = [None] * sensor_data.shape[0]

for i in range(len(sensor_data)):
    x_, y_ = splitSequence(sensor_data[i], look_back)

    if shuffle:
        x_,y_, p = shuffle_arrays(x_, y_)
        permutations[i] = p

    # print(f"x_ shape: {x_.shape}")

    x_train[i] = x_[:num_train]
    y_train[i] = y_[:num_train]
    x_val[i] = x_[num_train:num_train+num_val]
    y_val[i] = y_[num_train:num_train+num_val]
    x_test[i] = x_[num_train+num_val:]
    y_test[i] = y_[num_train+num_val:]

    if i == 0:
        print(f"sensordata shape: {sensor_data[i].shape}")
        print(f"x_ shape: {x_.shape}")
        print(f"y_ shape: {y_.shape}")
        print(f"x_train shape: {x_train[i].shape}")
        print(f"y_train shape: {y_train[i].shape}")

# x = np.array(x)
# y = np.array(y)
permutations = np.array(permutations)
# x, y = splitSequence(data, look_back)
# x, y = splitSequence(sensor_data, look_back)
# print(sensor_data.shape)
# print(x.shape)
# print(y.shape)

# if shuffle:
#     idx = np.random.permutation(len(x))
#     x,y = x[idx], y[idx]



# x_train, y_train = x[:num_train], y[:num_train]
# x_val, y_val = x[num_train:num_train + num_val], y[num_train:num_train + num_val]
# x_test, y_test = x[num_train + num_val:], y[num_train + num_val:]

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")





def zero_out_sequences(arr, percentage):
    # Calculate total values to remove
    total_values = len(arr)

    # Calculate number of sequences to remove, each of size 10
    num_sequences = total_values // 10
    remaining_values = total_values % 10

    sequences_to_remove = int(num_sequences * percentage)

    # Generate possible start indices for sequences
    indices = np.arange(num_sequences) * 10
    indices = indices.tolist()


    # print("indices: ", len(indices))
    # Select start indices randomly, without replacement
    start_indices = random.sample(indices, sequences_to_remove)

    # print("start_indices: ", len(start_indices))

    # Set the selected sequences to zero
    for start_index in start_indices:
        arr[start_index : start_index + 10] = np.nan

    # Remove the remaining values after sequences of 10
    if remaining_values > 0:
        arr[-remaining_values:] = np.nan

    num_removed = len(start_indices)*10 + remaining_values

    return arr, num_removed

# Test the function


def breakData(x, y, remove_amount = 0.0):
    print(x.shape)
    print(y.shape)

    # create single array from x and y
    arr = mergeSequences(x, y)
    length = arr.shape[1]
    newLength = 0

    # print("arr shape: ", arr.shape)

    # remove data
    for i in range(len(arr)):

        # remove single points
        # for j in range(toRemove):
        #     index = np.random.randint(0, length)
        #     arr[i][index] = np.nan

        # print("i = ", i)
        # print("  nans: ", np.count_nonzero(np.isnan(arr[i])))
        # remove sequences
        arr[i], removed = zero_out_sequences(arr[i], remove_amount)
        newLength = length - removed

        # print("  nans: ", np.count_nonzero(np.isnan(arr[i])))
        # print("  removed: ", removed)
        # print("  newLength: ", newLength)
        # while toRemove > 0:
        #     index = np.random.randint(0, length)
        #     removeLength = int(np.random.normal(meanRemoveLength, stdDev))
        #     arr[i][index:index+removeLength] = np.nan
        #     toRemove -= removeLength

    # print("x_t: ", np.count_nonzero(np.isnan(arr[0])))

    # recalculate data
    ## nan to 0
    arr = np.nan_to_num(arr)

    ## interpolate
    # for i in range(len(arr)):
    #     arr[i] = pd.Series(arr[i]).interpolate(method="linear", limit_direction="both").to_numpy()
    newLength = length
    # (print("YO"))
    # print(newLength)
    # print(np.count_nonzero(np.isnan(arr[0])))
    # print(arr[0][:100])


    ## remove nan
    # new_arr = np.empty((arr.shape[0], newLength))
    # for i in range(len(arr)):
    #     new_arr[i] = arr[i][~np.isnan(arr[i])]

    # print("new_arr shape: ", new_arr.shape)

    # print(arr[0].shape)
    # split the array into x and y again
    assert newLength - look_back > 0
    # print("x shape: ", x.shape)
    x_new = np.empty((x.shape[0], newLength - look_back, look_back))
    y_new = np.empty((y.shape[0], newLength - look_back))
    # print("x_new shape: ", x_new.shape)
    for i in range(len(arr)):
        x_new[i], y_new[i] = splitSequence(arr[i], look_back)


    return x_new, y_new

rmses = []
val_rmses = []
trainscores = []
testscores = []
# runs = x_train.shape[0]
# remove_amounts = [0.0]

remove_amounts = np.arange(0.0, 0.96, 0.05)
print(remove_amounts)
# remove_amounts.reverse()
runs = len(remove_amounts)
epochs = 100

# print("x_train")
# print(x_train[1])
# nx, ny = breakData(x_train, y_train, remove_amounts[0])
# print("broken")
# print(nx[1])

s = 53


# for i in range(runs):
for i in range(runs):
    #Add params to do optimizing at the top
    input_dim = 1
    # input_dim = data.shape[1]
    units = 60
    output_size = 1
    # output_size = y_train.shape[1]

    input = keras.Input((look_back, input_dim))
    #return sequences is necessary for sequential LSTM layers
    lstm1 = LSTM(units, return_sequences=True)(input)
    lstm2 = LSTM(units)(lstm1)
    out = Dense(output_size)(lstm2)
    model = keras.models.Model(inputs=input, outputs=out)
    # model.summary()

    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=0.0001),
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    cback = [keras.callbacks.EarlyStopping(patience=10)]
    # cback =[]
    # if runs == 1:
    #     cback = [keras.callbacks.EarlyStopping(patience=10)]
    print("x_train: ", np.count_nonzero(np.isnan(x_train[s])))
    x_t, y_t = breakData(x_train, y_train, remove_amounts[i])

    history = model.fit(
        x=x_t[s],
        y=y_t[s],
        validation_data=(x_val[s], y_val[s]),
        epochs=epochs,
        #makes the training stop early if it notices no improvements on the validation set 10 times in a row, to prevent overfitting
        callbacks=cback,
    )

    # save data to calculate the learning curve
    rmses.append(history.history['root_mean_squared_error'])
    val_rmses.append(history.history['val_root_mean_squared_error'])

    # make predictions
    trainPredict = model.predict(x_t[s])
    testPredict = model.predict(x_test[s])
    # invert predictions
    trainPredict = scalers[s].inverse_transform(trainPredict.reshape(-1,1))
    trainY = scalers[s].inverse_transform(y_t[s].reshape(-1,1))
    testPredict = scalers[s].inverse_transform(testPredict.reshape(-1,1))
    testY = scalers[s].inverse_transform(y_test[s].reshape(-1,1))
    # calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
    print(f'  Run: {i+1} of {runs} removeAmount: {remove_amounts[i]}')
    print(f'    Train Score: {trainScore:.2f} RMSE')
    testScore = np.sqrt(mean_squared_error(testY, testPredict))
    print(f'    Test Score: {testScore:.2f} RMSE')
    trainscores.append(trainScore)
    testscores.append(testScore)

for i in range(runs):
    print(f'Run {i+1}:')
    print(f'  Train Score: {trainscores[i]} RMSE')
    print(f'  Test Score: {testscores[i]} RMSE')

# rmses = scaler.inverse_transform(rmses)
# val_rmses = scaler.inverse_transform(val_rmses)

print("trainscores: " + str(trainscores))
print("testscores: " + str(testscores))
print("xt shape: " + str(x_t.shape))
print("xt: " + str(x_t[s][0]) )

# rmses = np.matrix(rmses)
# val_rmses = np.matrix(val_rmses)
#
# print(rmses.shape)
#
# rmse_avg = np.mean(rmses, axis=0).transpose()
# val_rmse_avg = np.mean(val_rmses, axis=0).transpose()
#
# print(rmse_avg.shape)
#
# rmse_std = np.std(rmses, axis=0).transpose()
# val_rmse_std = np.std(val_rmses, axis=0).transpose()
#
sigma = 1
skip = 3
#
# rmse_std_high = rmse_avg + rmse_std * sigma
# rmse_std_low = rmse_avg - rmse_std * sigma
# val_rmse_std_high = val_rmse_avg + val_rmse_std * sigma
# val_rmse_std_low = val_rmse_avg - val_rmse_std * sigma


# print(val_rmses)

# plt.plot(rmse_avg[skip:], label='train', color='orange')
# plt.plot(val_rmse_avg[skip:], label='validation', color='green')
# plt.plot(rmse_std_high[skip:], label='train std', linestyle='dashed', color='orange')
# plt.plot(rmse_std_low[skip:], label='_nolegend_', linestyle='dashed', color='orange')
# plt.plot(val_rmse_std_high[skip:], label='validation std', linestyle='dashed', color='green')
# plt.plot(val_rmse_std_low[skip:], label='_nolegend_', linestyle='dashed', color='green')
plt.plot(rmses[-1][skip:], label='train', color='orange')
plt.plot(val_rmses[-1][skip:], label='validation', color='green')
plt.title("learning curve")
plt.xlabel('epoch')
plt.ylabel('loss (RMSE)')
plt.legend(['train', 'validation', 'train_std', 'validation_std'], loc='upper right')
plt.show()


start = 0
end = 20

# y_predicted = model.predict(x_test[s])
# y_pred = scalers[s].inverse_transform(y_predicted.reshape(-1,1))
# y_testt = scalers[s].inverse_transform(y_test[s].reshape(-1,1))
# plt.plot(y_pred[start:end], label='predicted')
# plt.plot(y_testt[start:end], label='actual')

y_predicted = model.predict(x_t[s])

y_pred = scalers[s].inverse_transform(y_predicted.reshape(-1,1))
y_broken = scalers[s].inverse_transform(y_t[s].reshape(-1,1))
y_actual = scalers[s].inverse_transform(y_train[s].reshape(-1,1))

plt.plot(y_pred[start:end], label='predicted')
plt.plot(y_broken[start:end], label='training data')
plt.plot(y_actual[start:end], label='actual')



# if shuffle:
# l = len(y_predicted)
# y1 = y_train.append(y_val).append(y_test)
# y2 = y_train.append(y_val).append(y_test)
# y_predicted, y_test = unshuffle_arrays(y_predicted, y_test, permutations[0])


plt.title("predicted vs actual")
plt.xlabel('timestep')
plt.ylabel('traffic volume')
plt.legend(loc='upper right')
plt.show()

# print(x_train[0])
# sequence_drop = [5.937290703361259, 6.050255907588749, 6.113564393383473, 6.300986722461655, 6.306093783352975, 6.859348969027009, 6.425697389114244, 6.240484308133876, 6.4600056638554655, 6.46977843684363, 6.281919620840296, 6.425109767102989, 6.51481501083175, 6.33793144660578, 7.880060735156781, 11.660277875047855, 11.063649335792908, 7.419221222670605, 8.717067946557489]
remove_amounts = np.arange(0.0, 0.96, 0.05)

sequence_drop = [5.80275512333168, 6.097074039240171, 6.118148666724612, 6.138588157838037, 6.096692465094835, 6.147706809717529, 6.283660852507075, 6.318140090568146, 6.3330610615123994, 6.2887496825848475, 7.904956764473474, 6.499192069917341, 6.360463804026401, 9.159791759261122, 6.6706346903503535, 7.076429790541205, 7.197807768056538, 7.869420648871733, 9.065270828199019, 9.529965726389672]
sequence_drop2 = [5.923651939144846, 6.035728002472713, 6.1791228654829835, 6.139826990484982, 6.167036502114471, 6.180019359072112, 6.064523755903508, 6.354595843553823, 6.50349913792068, 6.274009976445937, 7.457652026070844, 6.449679928502428, 6.537263032230999, 6.730683428683406, 6.489970766991194, 6.879670540922892, 7.275912532737945, 9.623438339045531, 9.384346531674423, 12.000172337139487]

sequence_interpolation = [5.884398454724488, 5.893290210135884, 5.87991387844399, 5.93462353035117, 6.008400899349778, 6.098383739489963, 6.091886183628113, 6.186951920715532, 6.035041031627119, 6.213717222452649, 6.249146686296364, 6.2527202070611905, 6.358993455839185, 6.413530471994709, 6.315528279962114, 6.5055583889961035, 6.435025055606525, 6.455177000967018, 7.122407485830768, 6.639018064120929]

sequence_zero = [5.934568758521251, 6.039277001188321, 6.072149123289692, 6.075973408218182, 6.131611981291466, 6.279242710142609, 6.061097328583546, 6.222486422851454, 6.292696373500031, 6.22758849633677, 6.172963397342614, 6.344095689395306, 6.411802320541025, 6.87741220017679, 6.40915742020267, 6.65223040954923, 6.693940805894252, 6.415419592735293, 6.754215876477191, 7.934902211184823]

baseline = np.repeat(5.889696076280151, len(sequence_drop))

# plot testscores
print(testscores)
print(len(remove_amounts))
plt.plot(remove_amounts * 100, sequence_interpolation, label="interpolation")
plt.plot(remove_amounts * 100, sequence_drop, label="drop values")
plt.plot(remove_amounts * 100, sequence_zero, label="set to zero")
plt.plot(remove_amounts * 100, baseline, label="baseline", linestyle='dashed')
plt.title("Model loss when removing data in sequences")
plt.xlabel('Percentage of removed data')
plt.ylabel('RMSE')
plt.legend()
plt.show()



