import mlp_reductor as ds

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class MLPRegressor:
    def __init__(self, hidden_layers=(64, 32), learning_rate=0.01, epochs=100, show_learning_process=True):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self.show_learning_process = show_learning_process
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def _initialize_parameters(self, n_features):
        layer_dims = [n_features] + list(self.hidden_layers) + [1]
        for i in range(1, len(layer_dims)):
            self.weights.append(np.random.randn(layer_dims[i-1], layer_dims[i]) * 0.01)
            self.biases.append(np.zeros((1, layer_dims[i])))
    
    def _forward_propagation(self, X):
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            if i < len(self.weights) - 1:
                activation = self._sigmoid(z)
            else:
                activation = z
                
            activations.append(activation)
            
        return activations, z_values
    
    def _backward_propagation(self, X, y, activations, z_values):
        m = X.shape[0]
        grads = {}
        L = len(self.weights)
        
        y = np.array(y).reshape(-1, 1)
        
        error = activations[-1] - y
        grads[f'dW{L}'] = (1/m) * np.dot(activations[-2].T, error)
        grads[f'db{L}'] = (1/m) * np.sum(error, axis=0, keepdims=True)
        
        for l in reversed(range(L-1)):
            error = np.dot(error, self.weights[l+1].T) * self._sigmoid_derivative(activations[l+1])
            grads[f'dW{l+1}'] = (1/m) * np.dot(activations[l].T, error)
            grads[f'db{l+1}'] = (1/m) * np.sum(error, axis=0)
            
        return grads
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        n_features = X.shape[1]
        self._initialize_parameters(n_features)
        
        for epoch in range(self.epochs):
            activations, z_values = self._forward_propagation(X)
            grads = self._backward_propagation(X, y, activations, z_values)
            
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * grads[f'dW{i+1}']
                self.biases[i] -= self.learning_rate * grads[f'db{i+1}']
            
            if self.show_learning_process:
                mse = mean_squared_error(y, activations[-1])
                print(f"Epoch {epoch}, MSE: {mse:.4f}")
    
    def predict(self, X):
        X = np.array(X)
        activations, _ = self._forward_propagation(X)
        return activations[-1].flatten()

def main():
    data, target = ds.get_data(False)
    
    X_train, X_test, y_train, y_test = train_test_split(
        data, np.array(target), test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    mlp = MLPRegressor(hidden_layers=(64, 32), learning_rate=0.01, epochs=200)
    mlp.fit(X_train_scaled, y_train)
    
    train_pred = mlp.predict(X_train_scaled)
    test_pred = mlp.predict(X_test_scaled)
    
    print("\nTraining MSE:", mean_squared_error(y_train, train_pred))
    print("Test MSE:", mean_squared_error(y_test, test_pred))
    
    print("\nПервые 10 предсказаний из тестовой выборки:")
    for i in range(10):
        print(f"Пример {i+1}: Предсказано {test_pred[i]:.1f}, Фактически {y_test[i]:.1f}")

if __name__ == "__main__":
    main()