import prepare_dataset as dataset

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

learning_data = dataset.get_data_for_heatmap()

plt.figure(figsize=(25, 10))

learning_data = dataset.exclude_high_corelated_variables(learning_data)

matrix = np.tril(learning_data.corr())
sns.heatmap(learning_data.corr(), annot=True, fmt=".2f", linecolor='black', linewidths=0.5, mask=matrix, square=False, cbar=False)

plt.xticks(rotation=90)

plt.show()
