import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import classification_report, confusion_matrix

import mlp_classifier as ds

class MLPClassifier:
    def __init__(self, hidden_layers=(64, 32), learning_rate=0.01, epochs=100, random_state=42):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.weights = []
        self.biases = []
        self.classes_ = None
    
    def _initialize_parameters(self, n_features, n_classes):
        np.random.seed(self.random_state)
        layer_dims = [n_features] + list(self.hidden_layers) + [n_classes]
        
        for i in range(1, len(layer_dims)):
            self.weights.append(np.random.randn(layer_dims[i-1], layer_dims[i]) * np.sqrt(2./layer_dims[i-1]))
            self.biases.append(np.zeros((1, layer_dims[i])))
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def _softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)
    
    def _forward_propagation(self, X):
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            if i < len(self.weights) - 1:
                activation = self._relu(z)
            else:
                activation = self._softmax(z)
                
            activations.append(activation)
            
        return activations, z_values
    
    def _compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-15)) / m
    
    def _backward_propagation(self, X, y_true, activations, z_values):
        m = X.shape[0]
        grads = {}
        L = len(self.weights)
        
        error = activations[-1] - y_true
        grads[f'dW{L}'] = np.dot(activations[-2].T, error) / m
        grads[f'db{L}'] = np.sum(error, axis=0) / m
        
        for l in reversed(range(L-1)):
            error = np.asarray(np.dot(error, self.weights[l+1].T)) * self._relu_derivative(activations[l+1])
            grads[f'dW{l+1}'] = np.dot(activations[l].T, error) / m
            grads[f'db{l+1}'] = np.sum(error, axis=0) / m
            
        return grads
    
    def fit(self, X, y):
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        self.encoder_ = OneHotEncoder(sparse_output=False)
        y_encoded = self.encoder_.fit_transform(y.reshape(-1, 1))
        self.classes_ = self.encoder_.categories_[0]
        
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        self._initialize_parameters(n_features, n_classes)
        
        for epoch in range(self.epochs):
            activations, z_values = self._forward_propagation(X)
            grads = self._backward_propagation(X, y_encoded, activations, z_values)
            
            self._update_parameters(grads)
            
            if epoch % 10 == 0:
                self._print_learning_status(X, y, y_encoded, epoch, activations)

    def _print_learning_status(self, X, y, y_encoded, epoch, activations):
        loss = self._compute_loss(y_encoded, activations[-1])
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    def _update_parameters(self, grads):
        for i in range(len(self.weights)):
            corrected_weight = self.learning_rate * grads[f'dW{i+1}']
            corrected_bias = self.learning_rate * grads[f'db{i+1}'].reshape(1, -1)
            self.weights[i] -= corrected_weight
            self.biases[i] -= corrected_bias
    
    def predict_proba(self, X):
        X = np.array(X, dtype=np.float32)
        activations, _ = self._forward_propagation(X)
        return activations[-1]
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

def main():

    data, target_column = ds.get_data(False)

    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target_column)
    num_classes = len(label_encoder.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(data, target_encoded, test_size=0.2, random_state=42)

    y_train_cat = y_train
    y_test_cat = y_test

    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPClassifier(
        hidden_layers=(64, 32),
        learning_rate=0.01,
        epochs=2000
    )
    mlp.fit(X_train, y_train_cat)
    
    y_pred = mlp.predict(X_test)
    y_proba = mlp.predict_proba(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test_cat, y_pred))
    print("\nFirst 5 predictions:")
    for i in range(5):
        print(f"Sample {i}: True={y_test_cat[i]}, Pred={y_pred[i]}, Proba={np.max(y_proba[i]):.2f}")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

if __name__ == "__main__":
    main()
