import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore")

DATA_PATH = "dataset.csv"

def basic_preprocess(data : pd.DataFrame) -> pd.DataFrame:
    processed = data
    missing_count = processed.isna().sum()
    if (missing_count.any()):
        processed = processed.fillna(processed.mean())
    duplicates_count = processed.duplicated().sum()
    if (duplicates_count.any()):
        processed = processed.drop_duplicates()
    processed = process_height(processed)
    processed = fix_float_values(processed)
    processed = pd.get_dummies(processed, columns=["family_history_with_overweight", "CAEC", "CALC", "MTRANS", "SCC", "SMOKE", "Gender", "FAVC"], dtype=float)
    return processed

def preprocess_for_importance_calculation(data : pd.DataFrame) -> pd.DataFrame:
    processed = basic_preprocess(data)
    processed["NObeyesdad"] = pd.factorize(processed["NObeyesdad"])[0]
    return processed

def preprocess_for_heatmap(data : pd.DataFrame) -> pd.DataFrame:
    processed = basic_preprocess(data)
    processed = pd.get_dummies(processed)
    return processed

def process_height(data : pd.DataFrame) -> pd.DataFrame:
    data["Height" ] = (data["Height"] * 100).astype(int)
    return data

def process_age(data : pd.DataFrame) -> pd.DataFrame:
    data["Age"] = np.floor(data["Age"])
    return data

def fix_float_values(data : pd.DataFrame) -> pd.DataFrame:
    for attribute in data.columns:
        if data[attribute].dtype == np.float64:
            data[attribute] = np.floor(data[attribute]).astype(int)
    return data

def exclude_high_corelated_variables(data : pd.DataFrame) -> pd.DataFrame:
    computed = data.drop("Gender_Male", axis=1)
    computed = computed.drop("family_history_with_overweight_no", axis=1)
    computed = computed.drop("FAVC_no", axis=1)
    computed = computed.drop("CAEC_Sometimes", axis=1)
    computed = computed.drop("SMOKE_no", axis=1)
    computed = computed.drop("SCC_no", axis=1)
    computed = computed.drop("CALC_no", axis=1)
    computed = computed.drop("MTRANS_Public_Transportation", axis=1)
    return computed

def get_raw_data() -> pd.DataFrame: 
    return pd.read_csv(DATA_PATH)

def get_data_for_heatmap() -> pd.DataFrame:
    learning_data = get_raw_data()
    learning_data = preprocess_for_heatmap(learning_data)
    return learning_data

def get_data_for_importance_calculation() -> pd.DataFrame:
    learning_data = get_raw_data()
    learning_data = preprocess_for_importance_calculation(learning_data)
    return learning_data

def get_preprocessed_data() -> pd.DataFrame:
    data = get_raw_data()
    return basic_preprocess(data)
