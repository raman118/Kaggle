import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize a simple 3-layer neural network (input -> hidden -> output)
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output neurons
            learning_rate: Learning rate for gradient descent
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values (Xavier initialization)
        # W1: weights from input to hidden layer (input_size x hidden_size)
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        
        # b1: biases for hidden layer (1 x hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        
        # W2: weights from hidden to output layer (hidden_size x output_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        
        # b2: biases for output layer (1 x output_size)
        self.b2 = np.zeros((1, self.output_size))
        
        # Store values for backward propagation
        self.cache = {}
        
        # Store loss history for plotting
        self.loss_history = []
    
    def sigmoid(self, z):
        """
        Sigmoid activation function: 1 / (1 + e^(-z))
        Outputs values between 0 and 1
        """
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """
        Derivative of sigmoid function: sigmoid(z) * (1 - sigmoid(z))
        """
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def relu(self, z):
        """
        ReLU activation function: max(0, z)
        """
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """
        Derivative of ReLU: 1 if z > 0, else 0
        """
        return (z > 0).astype(float)
    
    def forward_propagation(self, X):
        """
        Forward propagation through the network
        
        Args:
            X: Input data (batch_size x input_size)
            
        Returns:
            A2: Final output (batch_size x output_size)
        """
        # Input layer to hidden layer
        # Z1 = X * W1 + b1 (linear transformation)
        Z1 = np.dot(X, self.W1) + self.b1
        
        # Apply activation function to get hidden layer output
        A1 = self.relu(Z1)  # Using ReLU for hidden layer
        
        # Hidden layer to output layer
        # Z2 = A1 * W2 + b2 (linear transformation)
        Z2 = np.dot(A1, self.W2) + self.b2
        
        # Apply activation function to get final output
        A2 = self.sigmoid(Z2)  # Using sigmoid for output layer
        
        # Store intermediate values for backward propagation
        self.cache = {
            'X': X,
            'Z1': Z1,
            'A1': A1,
            'Z2': Z2,
            'A2': A2
        }
        
        return A2
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute mean squared error loss
        
        Args:
            y_true: True labels (batch_size x output_size)
            y_pred: Predicted values (batch_size x output_size)
            
        Returns:
            loss: Mean squared error
        """
        m = y_true.shape[0]  # Number of examples
        loss = (1/(2*m)) * np.sum((y_pred - y_true)**2)
        return loss
    
    def backward_propagation(self, y_true):
        """
        Backward propagation to compute gradients
        
        Args:
            y_true: True labels (batch_size x output_size)
        """
        # Get cached values from forward propagation
        X = self.cache['X']
        Z1 = self.cache['Z1']
        A1 = self.cache['A1']
        Z2 = self.cache['Z2']
        A2 = self.cache['A2']
        
        m = X.shape[0]  # Number of examples
        
        # =============== BACKWARD PROPAGATION STEP BY STEP ===============
        
        # Step 1: Compute gradient of loss with respect to output (A2)
        # For MSE loss: dL/dA2 = (A2 - y_true)
        dA2 = A2 - y_true
        print(f"dA2 shape: {dA2.shape}")
        
        # Step 2: Compute gradient with respect to Z2 (before output activation)
        # Chain rule: dL/dZ2 = dL/dA2 * dA2/dZ2
        # Since A2 = sigmoid(Z2), dA2/dZ2 = sigmoid_derivative(Z2)
        dZ2 = dA2 * self.sigmoid_derivative(Z2)
        print(f"dZ2 shape: {dZ2.shape}")
        
        # Step 3: Compute gradients for W2 and b2
        # dL/dW2 = A1^T * dZ2 (derivative of Z2 = A1*W2 + b2 w.r.t W2)
        dW2 = (1/m) * np.dot(A1.T, dZ2)
        print(f"dW2 shape: {dW2.shape}")
        
        # dL/db2 = mean of dZ2 across examples
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        print(f"db2 shape: {db2.shape}")
        
        # Step 4: Compute gradient with respect to A1 (hidden layer output)
        # Chain rule: dL/dA1 = dZ2 * W2^T
        dA1 = np.dot(dZ2, self.W2.T)
        print(f"dA1 shape: {dA1.shape}")
        
        # Step 5: Compute gradient with respect to Z1 (before hidden activation)
        # Chain rule: dL/dZ1 = dL/dA1 * dA1/dZ1
        # Since A1 = ReLU(Z1), dA1/dZ1 = relu_derivative(Z1)
        dZ1 = dA1 * self.relu_derivative(Z1)
        print(f"dZ1 shape: {dZ1.shape}")
        
        # Step 6: Compute gradients for W1 and b1
        # dL/dW1 = X^T * dZ1
        dW1 = (1/m) * np.dot(X.T, dZ1)
        print(f"dW1 shape: {dW1.shape}")
        
        # dL/db1 = mean of dZ1 across examples
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        print(f"db1 shape: {db1.shape}")
        
        # Store gradients
        self.gradients = {
            'dW2': dW2,
            'db2': db2,
            'dW1': dW1,
            'db1': db1
        }
        
        return self.gradients
    
    def update_parameters(self):
        """
        Update parameters using gradient descent
        """
        # Update weights and biases using computed gradients
        self.W2 -= self.learning_rate * self.gradients['dW2']
        self.b2 -= self.learning_rate * self.gradients['db2']
        self.W1 -= self.learning_rate * self.gradients['dW1']
        self.b1 -= self.learning_rate * self.gradients['db1']
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the neural network
        
        Args:
            X: Training data (batch_size x input_size)
            y: Training labels (batch_size x output_size)
            epochs: Number of training iterations
            verbose: Print training progress
        """
        for epoch in range(epochs):
            # Forward propagation
            y_pred = self.forward_propagation(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Backward propagation
            if verbose and epoch == 0:
                print("=== DETAILED BACKWARD PROPAGATION (First Epoch) ===")
            self.backward_propagation(y)
            
            # Update parameters
            self.update_parameters()
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    def predict(self, X):
        """
        Make predictions on new data
        """
        return self.forward_propagation(X)
    
    def plot_loss(self):
        """
        Plot training loss over epochs
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

# ====================== EXAMPLE USAGE ======================

if __name__ == "__main__":
    # Create a simple dataset for XOR problem
    # XOR is not linearly separable, so it's a good test for neural networks
    np.random.seed(42)
    
    # XOR dataset
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    print("=== TRAINING DATA ===")
    print("Input X:")
    print(X)
    print("\nTarget y:")
    print(y)
    
    # Create and train the neural network
    print("\n=== CREATING NEURAL NETWORK ===")
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
    
    print(f"Initial W1 shape: {nn.W1.shape}")
    print(f"Initial b1 shape: {nn.b1.shape}")
    print(f"Initial W2 shape: {nn.W2.shape}")
    print(f"Initial b2 shape: {nn.b2.shape}")
    
    print("\n=== TRAINING ===")
    nn.train(X, y, epochs=1000, verbose=True)
    
    print("\n=== FINAL PREDICTIONS ===")
    predictions = nn.predict(X)
    print("Predictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]} -> Prediction: {predictions[i][0]:.4f}, Target: {y[i][0]}")
    
    # Plot training loss
    nn.plot_loss()
    
    print("\n=== DETAILED NETWORK INSPECTION ===")
    print(f"Final W1:\n{nn.W1}")
    print(f"Final b1:\n{nn.b1}")
    print(f"Final W2:\n{nn.W2}")
    print(f"Final b2:\n{nn.b2}")