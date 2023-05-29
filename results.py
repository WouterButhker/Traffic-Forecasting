import re

import matplotlib.pyplot as plt
import pandas as pd

with open("search.txt", "r") as f:
    content = f.read()

train_score_pattern = r'Train Score: ([\d.]+) RMSE'
test_score_pattern = r'Test Score: ([\d.]+) RMSE'

train_scores = re.findall(train_score_pattern, content)
test_scores = re.findall(test_score_pattern, content)


print(train_scores)


batch_size = 10
num_batches = min(len(train_scores), len(test_scores)) // batch_size + 1

data = []

for i in range(num_batches):
    start_index = i * batch_size
    end_index = start_index + batch_size
    batch_train_scores = train_scores[start_index:end_index]
    batch_test_scores = test_scores[start_index:end_index]
    batch_train_mean = sum(float(score) for score in batch_train_scores) / len(batch_train_scores)
    batch_test_mean = sum(float(score) for score in batch_test_scores) / len(batch_test_scores)
    data.append({
        'Batch': f"{(i * 4 + 1) *15}",
        'Train Scores': batch_train_scores,
        'Test Scores': batch_test_scores,
        'Train Mean': batch_train_mean,
        'Test Mean': batch_test_mean
    })

df = pd.DataFrame(data)


# Plotting the means
plt.plot(df['Batch'], df['Train Mean'], label='Train Mean')
plt.plot(df['Batch'], df['Test Mean'], label='Test Mean')

plt.xlabel('Minutes')
plt.ylabel('RMSE')
plt.title('RMSE per Batch for 1 sensor over 10 batches')
plt.legend()

plt.show()