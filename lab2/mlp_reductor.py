import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd

from lab1 import prepare_dataset as ds
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
    

def run_learning(learning_rate=0.01, remove_high_corelated_variables=False, report_result=False, plot_graphics=False, show_learning_process=True):
    data, target_column = get_data(remove_high_corelated_variables)
    
    X_train, X_test, y_train, y_test = train_test_split(data, target_column, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                loss='mean_squared_error',
                metrics=['mae'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    history = model.fit(X_train, y_train_scaled,
                       epochs=100,
                       batch_size=32,
                       validation_split=0.2,
                       callbacks=[early_stopping],
                       verbose=1 if show_learning_process else 0)
    
    if report_result:
        show_report(X_test, y_test_scaled, model, target_scaler)
    
    if plot_graphics:
        visualise(history)
    
    return model, scaler, target_scaler

def show_report(X_test, y_test_scaled, model, target_scaler):
    y_pred_scaled = model.predict(X_test)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_test = target_scaler.inverse_transform(y_test_scaled)
    
    test_loss, test_mae = model.evaluate(X_test, y_test_scaled, verbose=0)
    print(f"\nTest Loss (MSE): {test_loss:.4f}\nTest MAE: {test_mae:.4f}")
    
    print("\nПервые 10 тестовых примеров:")
    print("{:<10} {:<15} {:<15}".format("Пример", "Предсказание", "Актуальное"))
    print("-" * 40)
    
    for i in range(10):
        print("{:<10} {:<15.2f} {:<15.2f}".format(i+1, y_pred[i][0], y_test[i][0]))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:50], y_pred[:50])
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Actual Weight')
    plt.ylabel('Predicted Weight')
    plt.title('Actual vs Predicted Weight')
    plt.show()

def get_data(remove_high_corelated_variables):
    TARGET = 'Weight'

    data = ds.get_preprocessed_data()
    data = pd.get_dummies(data, columns=["NObeyesdad"])
    data = ds.exclude_high_corelated_variables(data) if remove_high_corelated_variables else data
    target_column = data[TARGET]
    data = data.drop(TARGET, axis=1)
    return data, target_column


def visualise(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Mean Squared Error over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model, scaler, label_encoder = run_learning(learning_rate=0.001, plot_graphics=True, report_result=True, remove_high_corelated_variables=False)
