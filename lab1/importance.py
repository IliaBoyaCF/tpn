import prepare_dataset as dataset

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def split_info(data : pd.DataFrame, attr_name : str) -> float:
    counts = data[attr_name].value_counts()

    result = 0.0
    total = np.sum(counts)
    
    for count in counts:
        probability = count / total
        result -= probability * np.log2(probability)

    return result

def entropy(s : pd.Series):
    value_counts = s.value_counts()
    probabilities = value_counts / value_counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(data : pd.DataFrame, target : pd.DataFrame, attr_name : str) -> float:
    total_entropy = entropy(target)
    values, counts = np.unique(data[attr_name], return_counts=True)

    weighted_entropy = 0
    total_count = np.sum(counts)
    for value, count in zip(values, counts):
        subset = target[data[attr_name] == value]
        weighted_entropy += (count / total_count) * entropy(subset)
    return total_entropy - weighted_entropy

def manual_importance_calculation(data : pd.DataFrame, target : str) -> dict:
    results = {}

    for attribute in data.columns:
        if (attribute == target):
            continue
        results[attribute] = information_gain(data, data[target], attribute) / split_info(data, attribute)

    result = {k : v for k, v in sorted(results.items(), key=lambda item : item[1], reverse=True)}
    return pd.DataFrame({"Feature" : pd.Index(result.keys(), dtype='object'), "Importance" : result.values()})

def library_importance_calculation(data):
    X = data.loc[:, data.columns != "NObeyesdad"]
    y = data["NObeyesdad"]

    dtree = DecisionTreeClassifier(random_state=42)
    dtree.fit(X, y)

    importances = dtree.feature_importances_

    imp_df = pd.DataFrame({"Feature" : X.columns, "Importance" : importances})
    imp_df = imp_df.sort_values(by="Importance", ascending=False)
    return imp_df

data = dataset.get_data_for_importance_calculation()

data = dataset.exclude_high_corelated_variables(data)

manual = manual_importance_calculation(data, 'NObeyesdad')
print("---------Manual importance calculation----------")
print(manual)

library = library_importance_calculation(data)
print("---------Library-implemented importance calculation----------")
print(library)
