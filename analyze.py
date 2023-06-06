import os

import matplotlib.pyplot as plt
import pandas as pd

file_path = os.path.join(os.path.dirname(__file__), "results/4_steps_all_intersections")

dfs = []

for file in os.listdir(file_path):
    # fetch file and append it to df
    # Column might already exists, in that case append, otherwise create new column
    if file.endswith(".csv") and "results" in file:
        df = pd.read_csv(file_path + "/" + file)
        dfs.append(df)

# Columns might already exists, in that case append, otherwise create new column
df = pd.concat(dfs, axis=1)
# df = df.groupby(by=df.columns, axis=1).mean()

means = df.mean(axis=0)

df_train = df[[col for col in df.columns if "train_score" in col]]
df_test = df[[col for col in df.columns if "test_score" in col]]

df_relative_train = df[[col for col in df.columns if "train_relative" in col]]
df_relative_test = df[[col for col in df.columns if "test_relative" in col]]

# plt.plot(df_train, label='train')
# plt.plot(df_test, label='test')
# plt.legend()
# plt.show()

import matplotlib.pyplot as plt

# set the figure size
plt.figure(figsize=(10, 6))

df_train = df_train.reindex(
    sorted(df_train.columns, key=lambda x: int(x.split("_")[-1])), axis=1
)
df_test = df_test.reindex(
    sorted(df_test.columns, key=lambda x: int(x.split("_")[-1])), axis=1
)

# calculate mean scores for train data and plot
train_times = [(int(col.split("_")[-1]) / 4) for col in df_train.columns]
train_scores = df_train.mean().values
plt.plot(train_times, train_scores, label="Train", marker="o")

# calculate mean scores for test data and plot
test_times = [(int(col.split("_")[-1]) / 4) for col in df_test.columns]
test_scores = df_test.mean().values
plt.plot(test_times, test_scores, label="Test", marker="o")

train_relative_scores = df_relative_train.mean().values
plt.plot(train_times, train_relative_scores, label="Train relative", marker="o")

test_relative_scores = df_relative_test.mean().values
plt.plot(test_times, test_relative_scores, label="Test relative", marker="o")
# add labels and title
plt.xlabel("Future time (in hours)")
plt.ylabel("RMSE score")
plt.title("Mean scores for different time steps")
plt.legend()

# show plot
plt.show()
