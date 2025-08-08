import numpy as np
import matplotlib.pyplot as plt

class SimpleGradientDescent:
    """
    Simple Gradient Descent for Linear Regression
    
    Finds the best line (y = mx + b) through data points by:
    1. Starting with random line
    2. Measuring how wrong it is
    3. Adjusting the line to be less wrong
    4. Repeat until good enough
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        """
        Set up the learning process
        
        learning_rate: How big steps to take (0.01 = small careful steps)
        max_iterations: How many tries before giving up
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.cost_history = []  # Track how we improve over time
        
    def compute_cost(self, X, y, theta):
        """
        Calculate how wrong our predictions are (Mean Squared Error)
        
        Think of it as: "On average, how far off are my guesses?"
        
        X: input data with bias column added
        y: correct answers  
        theta: current line parameters [bias, slope]
        """
        m = len(y)  # number of data points
        predictions = X @ theta  # make predictions with current line
        errors = predictions - y  # how far off each prediction is
        cost = (1/(2*m)) * np.sum(errors**2)  # average squared error
        return cost
    
    def compute_gradients(self, X, y, theta):
        """
        Figure out which direction to adjust our line
        
        Returns: how much to change each parameter
        - If gradient is positive: decrease parameter
        - If gradient is negative: increase parameter
        """
        m = len(y)
        predictions = X @ theta  # current predictions
        errors = predictions - y  # how wrong we are
        gradients = (1/m) * X.T @ errors  # math to find adjustment direction
        return gradients
    
    def fit(self, X, y):
        """
        Learn the best line through the data
        
        X: input features (like house size)
        y: target values (like house price)
        """
        # Add column of 1's for y-intercept (bias term)
        # This lets our line cross y-axis anywhere, not just at (0,0)
        X = np.column_stack([np.ones(X.shape[0]), X])
        
        # Start with line at origin (bias=0, slope=0)
        self.theta = np.zeros(X.shape[1])  
        
        print(f"Starting training with {len(y)} data points...")
        
        # Try to improve the line many times
        for i in range(self.max_iterations):
            # Step 1: See how wrong we currently are
            cost = self.compute_cost(X, y, self.theta)
            self.cost_history.append(cost)
            
            # Step 2: Calculate which way to adjust
            gradients = self.compute_gradients(X, y, self.theta)
            
            # Step 3: Adjust our line parameters
            # We subtract because gradients point "uphill" but we want to go "downhill"
            self.theta = self.theta - self.learning_rate * gradients
            
            # Print progress occasionally
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")
        
        print(f"Training finished! Final cost: {cost:.4f}")
        return self
    
    def predict(self, X):
        """
        Make predictions with our learned line
        
        X: new input data
        Returns: predicted values
        """
        # Add bias column (same as in training)
        X = np.column_stack([np.ones(X.shape[0]), X])
        return X @ self.theta
    
    def plot_results(self, X, y):
        """
        Show two plots:
        1. How cost decreased during training
        2. Our final line vs the actual data
        """
        # Plot 1: Training progress
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.cost_history)
        plt.title('How Cost Decreased During Training')
        plt.xlabel('Iteration')
        plt.ylabel('Cost (Lower = Better)')
        plt.grid(True)
        
        # Plot 2: Final results (only works for 1D input)
        if X.shape[1] == 1:
            plt.subplot(1, 2, 2)
            
            # Plot actual data points
            plt.scatter(X, y, color='blue', alpha=0.6, label='Actual Data')
            
            # Plot our learned line
            predictions = self.predict(X)
            # Sort for smooth line
            sorted_idx = np.argsort(X.ravel())
            plt.plot(X[sorted_idx], predictions[sorted_idx], 
                    'red', linewidth=2, label='Our Line')
            
            plt.title('Our Line vs Actual Data')
            plt.xlabel('X')
            plt.ylabel('y')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_line_equation(self):
        """
        Show the equation of our learned line
        """
        bias = self.theta[0]
        slope = self.theta[1] if len(self.theta) > 1 else 0
        print(f"Learned line: y = {slope:.3f}x + {bias:.3f}")
        return slope, bias


# Example: Let's test it!
if __name__ == "__main__":
    print("=== Simple Gradient Descent Demo ===")
    
    # Create fake data: y = 2x + 1 with some noise
    np.random.seed(42)
    X = np.random.uniform(0, 10, 50).reshape(-1, 1)  # 50 random X values
    y = 2 * X.ravel() + 1 + np.random.normal(0, 1, 50)  # y = 2x + 1 + noise
    
    print("True line: y = 2.0x + 1.0")
    print("Let's see if gradient descent can figure this out...\n")
    
    # Train the model
    model = SimpleGradientDescent(learning_rate=0.01, max_iterations=1000)
    model.fit(X, y)
    
    # See what we learned
    print()
    model.get_line_equation()
    
    # Show results
    model.plot_results(X, y)