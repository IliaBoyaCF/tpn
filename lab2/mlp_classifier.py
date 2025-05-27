import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd

from lab1 import prepare_dataset as ds
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
    

def run_learning(learning_rate=0.001, remove_high_corelated_variables=False, report_result=False, plot_graphics=False, show_learning_process=True):

    data, target_column = get_data(remove_high_corelated_variables)

    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target_column)
    num_classes = len(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(data, target_encoded, test_size=0.2, random_state=42)

    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(10, activation='relu'),
        Dropout(0.2),
        Dense(7, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])


    model.compile(optimizer=Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy'])


    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

    verbose = 1 if show_learning_process else 0

    history = model.fit(X_train, y_train_cat,
                        epochs=200,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stopping],
                        verbose=verbose)


    if (report_result):
        show_report(X_test, y_test, y_test_cat, model, target_names=label_encoder.classes_)

    
    if (plot_graphics):
        visualise(history)

    return (model, scaler, label_encoder)

def show_report(X_test, y_test, y_test_cat, model, target_names=None):
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}\nTest Loss: {test_loss:.4f}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=target_names));

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_classes)
    cm = pd.DataFrame(cm, index=target_names, columns=target_names) if target_names is not None else cm
    print(cm)

def get_data(remove_high_corelated_variables):
    TARGET = 'NObeyesdad'

    data = ds.get_preprocessed_data()
    data = ds.exclude_high_corelated_variables(data) if remove_high_corelated_variables else data
    target_column = data[TARGET]
    data = data.drop(TARGET, axis=1)
    return data,target_column


def visualise(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def predict(model, scaler, label_encoder, data):
    single_sample = data.iloc[0:1]
    single_sample_scaled = scaler.transform(single_sample)

    sample_proba = model.predict(single_sample_scaled)
    sample_class = np.argmax(sample_proba)
    sample_label = label_encoder.inverse_transform([sample_class])[0]

    print(f"final prediction: {sample_label}, class: {sample_class}")

model, scaler, label_encoder = run_learning(learning_rate=0.001, plot_graphics=True, report_result=True, remove_high_corelated_variables=True)

data = get_data(False)[0]
