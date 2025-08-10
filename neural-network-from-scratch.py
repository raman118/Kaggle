import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class Activation:
    @staticmethod
    def relu(x): return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x): return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x): return x * (1 - x)
    
    @staticmethod
    def tanh(x): return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x): return 1 - x**2

class Layer:
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        # Xavier/Glorot initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        
        # For momentum/Adam optimizer
        self.vw = np.zeros_like(self.weights)
        self.vb = np.zeros_like(self.biases)
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.z = inputs @ self.weights + self.biases
        
        if self.activation == 'relu':
            self.output = Activation.relu(self.z)
        elif self.activation == 'sigmoid':
            self.output = Activation.sigmoid(self.z)
        elif self.activation == 'tanh':
            self.output = Activation.tanh(self.z)
        else:
            self.output = self.z  # linear
            
        return self.output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Compute activation derivative
        if self.activation == 'relu':
            grad_z = grad_output * Activation.relu_derivative(self.z)
        elif self.activation == 'sigmoid':
            grad_z = grad_output * Activation.sigmoid_derivative(self.output)
        elif self.activation == 'tanh':
            grad_z = grad_output * Activation.tanh_derivative(self.output)
        else:
            grad_z = grad_output
            
        # Compute gradients
        self.grad_weights = self.inputs.T @ grad_z / self.inputs.shape[0]
        self.grad_biases = np.mean(grad_z, axis=0, keepdims=True)
        
        # Return gradient for previous layer
        return grad_z @ self.weights.T

class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], activations: Optional[List[str]] = None):
        self.layers = []
        self.loss_history = []
        
        if activations is None:
            activations = ['relu'] * (len(layer_sizes) - 2) + ['sigmoid']
            
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i + 1], activations[i])
            self.layers.append(layer)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, X: np.ndarray, y: np.ndarray, predictions: np.ndarray):
        # Start with loss gradient
        grad = predictions - y
        
        # Backpropagate through layers
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def update_weights(self, learning_rate: float, momentum: float = 0.9):
        for layer in self.layers:
            # Momentum update
            layer.vw = momentum * layer.vw - learning_rate * layer.grad_weights
            layer.vb = momentum * layer.vb - learning_rate * layer.grad_biases
            
            layer.weights += layer.vw
            layer.biases += layer.vb
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Binary cross-entropy for classification, MSE for regression
        if y_true.shape[1] == 1 and np.all((y_true == 0) | (y_true == 1)):
            # Binary classification
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Prevent log(0)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # Regression
            return np.mean((y_true - y_pred) ** 2)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              learning_rate: float = 0.01, momentum: float = 0.9, 
              validation_split: float = 0.2, verbose: bool = True) -> dict:
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X_train)
            loss = self.compute_loss(y_train, predictions)
            
            # Backward pass
            self.backward(X_train, y_train, predictions)
            self.update_weights(learning_rate, momentum)
            
            # Validation
            if len(X_val) > 0:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                history['val_loss'].append(val_loss)
            
            history['loss'].append(loss)
            
            if verbose and epoch % 100 == 0:
                val_str = f", Val Loss: {val_loss:.4f}" if len(X_val) > 0 else ""
                print(f"Epoch {epoch}: Loss: {loss:.4f}{val_str}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        if y.shape[1] == 1:  # Binary classification
            pred_classes = (predictions > 0.5).astype(int)
        else:  # Multi-class
            pred_classes = np.argmax(predictions, axis=1, keepdims=True)
            y = np.argmax(y, axis=1, keepdims=True)
        return np.mean(pred_classes == y)

# Example Usage & Demos
def demo_xor():
    print("=== XOR Problem Demo ===")
    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    nn = NeuralNetwork([2, 8, 1], ['relu', 'sigmoid'])
    history = nn.train(X, y, epochs=2000, learning_rate=0.1, validation_split=0)
    
    print("\nResults:")
    predictions = nn.predict(X)
    for i in range(len(X)):
        print(f"Input: {X[i]} -> Output: {predictions[i][0]:.3f} (Target: {y[i][0]})")
    
    print(f"Accuracy: {nn.accuracy(X, y):.2%}")

def demo_spiral():
    print("\n=== Spiral Classification Demo ===")
    # Generate spiral data
    np.random.seed(42)
    N = 200  # points per class
    K = 2    # classes
    X = np.zeros((N*K, 2))
    y = np.zeros((N*K, 1))
    
    for j in range(K):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2
        X[ix] = np.column_stack((r*np.sin(t), r*np.cos(t)))
        y[ix] = j
    
    # Train network
    nn = NeuralNetwork([2, 32, 16, 1], ['relu', 'relu', 'sigmoid'])
    history = nn.train(X, y, epochs=1000, learning_rate=0.01)
    
    print(f"Final Accuracy: {nn.accuracy(X, y):.2%}")

if __name__ == "__main__":
    demo_xor()
    demo_spiral()