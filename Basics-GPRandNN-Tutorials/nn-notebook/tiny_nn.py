import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy


class Tensor(numpy.ndarray):
    """Custom tensor class inheriting from numpy.ndarray"""

    def __new__(cls, input_array, requires_grad=True):
        obj = numpy.asarray(input_array).view(cls)
        obj.requires_grad = requires_grad
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
                return
        self.requires_grad = getattr(obj, 'requires_grad', False)

    @property
    def grad(self):
        """Gradient of the tensor, initialized to zeros"""
        if self.requires_grad:
            if not hasattr(self, "_grad"):
                self._grad = numpy.zeros_like(self)
            return self._grad
        else:
            return None

    @grad.setter
    def grad(self, value):
        """Set the gradient of the tensor"""
        if self.requires_grad:
            if isinstance(value, numpy.ndarray):
                self._grad = value
            else:
                raise ValueError("Gradient must be a numpy array")
        else:
            pass

class Module:
    """Base class for all neural network modules"""

    def __init__(self):
        self.training = True
        self.params = []  # Will store parameters (weights, biases)

    def initialize(self, *args, **kwargs):
        """Initialize parameters - to be implemented by subclasses"""
        pass

    def forward(self, x):
        """Forward pass - to be implemented by subclasses"""
        raise NotImplementedError

    def backward(self, grad_output):
        """Backward pass - to be implemented by subclasses"""
        raise NotImplementedError

    def zero_grad(self):
        """Zero out gradients for all parameters"""
        for param in self.params:
            if hasattr(param, "grad"):
                param.grad = numpy.zeros_like(param.grad)

    def train(self):
        """Set module to training mode"""
        self.training = True

    def eval(self):
        """Set module to evaluation mode"""
        self.training = False

    @property
    def num_params(self):
        """Return total number of parameters in this module"""
        return sum(param.size for param in self.params if isinstance(param, Tensor))

class MSELoss(Module):
    """Mean Squared Error Loss"""

    def __init__(self):
        super().__init__()
        self.last_input = None
        self.last_target = None

    def forward(self, input, target):
        """
        Compute MSE loss
        Args:
            input: predicted values (batch_size, output_features)
            target: true values (batch_size, output_features)
        """
        self.last_input = input
        self.last_target = target
        diff = input - target
        loss = numpy.mean(diff**2)
        return loss

    def backward(self):
        """Compute gradient of MSE loss w.r.t. input"""
        batch_size = self.last_input.shape[0]
        grad = 2 * (self.last_input - self.last_target) / batch_size
        return grad


class MAELoss(Module):
    """Mean Absolute Error Loss"""

    def __init__(self):
        super().__init__()
        self.last_input = None
        self.last_target = None

    def forward(self, input, target):
        """
        Compute MAE loss
        Args:
            input: predicted values (batch_size, output_features)
            target: true values (batch_size, output_features)
        """
        self.last_input = input
        self.last_target = target
        diff = numpy.abs(input - target)
        loss = numpy.mean(diff)
        return loss

    def backward(self):
        """Compute gradient of MAE loss w.r.t. input"""
        batch_size = self.last_input.shape[0]
        grad = numpy.sign(self.last_input - self.last_target) / batch_size
        return grad


class ReLU(Module):
    """Rectified Linear Unit activation function"""

    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, x):
        """Apply ReLU: f(x) = max(0, x)"""
        self.last_input = x
        return numpy.maximum(0, x)

    def backward(self, grad_output):
        """Compute gradient: f'(x) = 1 if x > 0, else 0"""
        grad_input = grad_output * (self.last_input > 0).astype(float)
        return grad_input


class Sigmoid(Module):
    """Sigmoid activation function"""

    def __init__(self):
        super().__init__()
        self.last_output = None

    def forward(self, x):
        """Apply Sigmoid: f(x) = 1 / (1 + exp(-x))"""
        # Clip x to prevent overflow
        x_clipped = numpy.clip(x, -500, 500)
        output = 1 / (1 + numpy.exp(-x_clipped))
        self.last_output = output
        return output

    def backward(self, grad_output):
        """Compute gradient: f'(x) = f(x) * (1 - f(x))"""
        grad_input = grad_output * self.last_output * (1 - self.last_output)
        return grad_input


class Tanh(Module):
    """Hyperbolic tangent activation function"""

    def __init__(self):
        super().__init__()
        self.last_output = None

    def forward(self, x):
        """Apply Tanh: f(x) = tanh(x)"""
        output = numpy.tanh(x)
        self.last_output = output
        return output

    def backward(self, grad_output):
        """Compute gradient: f'(x) = 1 - tanh²(x)"""
        grad_input = grad_output * (1 - self.last_output**2)
        return grad_input


class Linear(Module):
    """Fully connected linear layer"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Initialize parameters
        self.initialize()

        # Store parameters for optimizer
        self.params = [self.weight]
        if self.use_bias:
            self.params.append(self.bias)

        # Store inputs for backward pass
        self.last_input = None

    def initialize(self, seed=None):
        """Initialize weights using Xavier initialization"""
        # Xavier initialization for better convergence
        if seed is not None:
            rng = numpy.random.RandomState(seed)
        else:
            rng = numpy.random.RandomState(42)  # Default seed

        limit = numpy.sqrt(6.0 / (self.in_features + self.out_features))
        self.weight = Tensor(rng.uniform(-limit, limit, (self.in_features, self.out_features)))
        self.weight.grad = numpy.zeros_like(self.weight)

        if self.use_bias:
            self.bias = Tensor(numpy.zeros((1, self.out_features)))
            self.bias.grad = numpy.zeros_like(self.bias)

    def forward(self, x):
        """
        Forward pass: y = x @ W + b
        Args:
            x: input tensor (batch_size, in_features)
        Returns:
            output tensor (batch_size, out_features)
        """
        self.last_input = x
        output = x @ self.weight
        if self.use_bias:
            output = output + self.bias
        return output

    def backward(self, grad_output):
        """
        Backward pass: compute gradients w.r.t. weights, bias, and input
        Args:
            grad_output: gradient from next layer (batch_size, out_features)
        Returns:
            grad_input: gradient w.r.t. input (batch_size, in_features)
        """
        # Gradient w.r.t. weight: X^T @ grad_output
        self.weight.grad += self.last_input.T @ grad_output

        # Gradient w.r.t. bias: sum over batch dimension
        if self.use_bias:
            self.bias.grad += numpy.sum(grad_output, axis=0, keepdims=True)

        # Gradient w.r.t. input: grad_output @ W^T
        grad_input = grad_output @ self.weight.T
        return grad_input


class Sequential(Module):
    """Sequential container for chaining modules"""

    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

        # Collect all parameters from submodules
        self.params = []
        for module in self.modules:
            if hasattr(module, "params"):
                self.params.extend(module.params)

    def initialize(self, seed=None):
        """Initialize all modules in the sequential container"""
        for module in self.modules:
            if hasattr(module, "initialize"):
                module.initialize(seed)

    def add_module(self, module):
        """Add a module to the sequential container"""
        self.modules.append(module)
        if hasattr(module, "params"):
            self.params.extend(module.params)

    def forward(self, x):
        """Forward pass through all modules sequentially"""
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, grad_output):
        """Backward pass through all modules in reverse order"""
        grad = grad_output
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad

    def zero_grad(self):
        """Zero gradients for all modules"""
        for module in self.modules:
            module.zero_grad()

    def train(self):
        """Set all modules to training mode"""
        self.training = True
        for module in self.modules:
            if hasattr(module, "train"):
                module.train()

    def eval(self):
        """Set all modules to evaluation mode"""
        self.training = False
        for module in self.modules:
            if hasattr(module, "eval"):
                module.eval()

    def save(self, path):
        """Save model parameters to file"""
        model_data = {"modules": [], "params": []}

        # Save module information and parameters
        for i, module in enumerate(self.modules):
            module_info = {"type": type(module).__name__, "params": {}}

            if isinstance(module, Linear):
                module_info["params"] = {
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "use_bias": module.use_bias,
                    "weight": module.weight.copy(),
                    "bias": module.bias.copy() if module.use_bias else None,
                }

            model_data["modules"].append(module_info)

        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load model parameters from file"""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        # Reconstruct modules with saved parameters
        for i, module_info in enumerate(model_data["modules"]):
            if module_info["type"] == "Linear":
                params = module_info["params"]
                module = self.modules[i]
                if isinstance(module, Linear):
                    module.weight = params["weight"].copy()
                    if params["bias"] is not None:
                        module.bias = params["bias"].copy()
                    # Reinitialize gradients
                    module.weight.grad = numpy.zeros_like(module.weight)
                    if module.use_bias:
                        module.bias.grad = numpy.zeros_like(module.bias)

        print(f"Model loaded from {path}")

    def visualize(self):
        """Visualize the model architecture"""
        print("Model Architecture:")
        for i, module in enumerate(self.modules):
            print(f"Module {i + 1}: {type(module).__name__}")
            if hasattr(module, "params"):
                for param in module.params:
                    print(f"  - Parameter: {param.shape} (requires_grad={hasattr(param, 'grad')})")
        print(f"Total parameters: {self.num_params}")


class SGD:
    """Stochastic Gradient Descent optimizer"""

    def __init__(self, params: List, lr: float = 0.01):
        self.params = params
        self.lr = lr
        self.loss_history = []
        self.step_count = 0

    def step(self):
        """Update parameters using gradients"""
        for param in self.params:
            if hasattr(param, "grad"):
                param -= self.lr * param.grad
        self.step_count += 1

    def zero_grad(self):
        """Zero out all parameter gradients"""
        for param in self.params:
            if hasattr(param, "grad"):
                param.grad = numpy.zeros_like(param.grad)

    def record_loss(self, loss):
        """Record loss for visualization"""
        self.loss_history.append(loss)

    def visualize(self, title="Training Loss", save_path=None):
        """Visualize training loss over time"""
        if not self.loss_history:
            print("No loss history to visualize")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, "b-", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.yscale("log")  # Log scale often better for loss visualization

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def get_final_loss(self):
        """Get the final loss value"""
        return self.loss_history[-1] if self.loss_history else None


# Test code and example usage
def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    """
    Compute Lennard-Jones potential: V(r) = 4ε[(σ/r)^12 - (σ/r)^6]
    Args:
        r: distance(s)
        epsilon: depth of potential well
        sigma: distance at which potential is zero
    """
    r_safe = numpy.maximum(r, sigma * 0.5)  # Avoid division by zero
    sigma_over_r = sigma / r_safe
    return 4 * epsilon * (sigma_over_r**12 - sigma_over_r**6)


def generate_lj_data(n_samples=1000, r_min=0.8, r_max=3.0, epsilon=1.0, sigma=1.0, noise=0.01):
    """Generate training data for LJ potential"""
    r = numpy.random.uniform(r_min, r_max, (n_samples, 1))
    V = lennard_jones_potential(r, epsilon, sigma)

    # Add some noise to make it more realistic
    if noise > 0:
        V += numpy.random.normal(0, noise * numpy.abs(V), V.shape)

    return r, V


def test_nn_lj_fitting():
    """Test the neural network by fitting a Lennard-Jones potential"""
    print("Testing Neural Network with Lennard-Jones Potential Fitting")
    print("=" * 60)

    # Generate training data
    print("Generating training data...")
    r_train, V_train = generate_lj_data(n_samples=2000, noise=0.02)
    r_test, V_test = generate_lj_data(n_samples=500, r_min=0.9, r_max=2.8, noise=0.0)

    print(f"Training data shape: {r_train.shape} -> {V_train.shape}")
    print(f"Test data shape: {r_test.shape} -> {V_test.shape}")

    # Create neural network
    print("\nBuilding neural network...")
    model = Sequential(
        Linear(1, 32),  # Input layer: 1D distance -> 32 neurons
        Tanh(),  # Activation
        Linear(32, 64),  # Hidden layer: 32 -> 64 neurons
        ReLU(),  # Activation
        Linear(64, 32),  # Hidden layer: 64 -> 32 neurons
        Tanh(),  # Activation
        Linear(32, 1),  # Output layer: 32 -> 1 (potential energy)
    )

    # Initialize optimizer and loss function
    optimizer = SGD(model.params, lr=0.001)
    criterion = MSELoss()

    print(f"Model has {len(model.params)} parameter tensors")

    # Training loop
    print("\nStarting training...")
    n_epochs = 1000
    batch_size = 64
    n_batches = len(r_train) // batch_size

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0

        # Shuffle data
        indices = numpy.random.permutation(len(r_train))
        r_shuffled = r_train[indices]
        V_shuffled = V_train[indices]

        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size

            # Get batch
            r_batch = r_shuffled[start_idx:end_idx]
            V_batch = V_shuffled[start_idx:end_idx]

            # Forward pass
            optimizer.zero_grad()
            V_pred = model.forward(r_batch)
            loss = criterion.forward(V_pred, V_batch)

            # Backward pass
            grad_loss = criterion.backward()
            model.backward(grad_loss)

            # Update parameters
            optimizer.step()
            epoch_loss += loss

        # Record average loss for this epoch
        avg_loss = epoch_loss / n_batches
        optimizer.record_loss(avg_loss)

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.6f}")

    print("Training completed!")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    V_test_pred = model.forward(r_test)
    test_loss = criterion.forward(V_test_pred, V_test)
    print(f"Test Loss (MSE): {test_loss:.6f}")

    # Calculate test MAE for comparison
    mae_criterion = MAELoss()
    test_mae = mae_criterion.forward(V_test_pred, V_test)
    print(f"Test Loss (MAE): {test_mae:.6f}")

    # Visualize results
    print("\nGenerating visualizations...")

    # Plot training loss
    optimizer.visualize("Training Loss - LJ Potential Fitting")

    # Plot LJ potential fit
    r_plot = numpy.linspace(0.8, 3.0, 200).reshape(-1, 1)
    V_true = lennard_jones_potential(r_plot)
    V_pred_plot = model.forward(r_plot)

    plt.figure(figsize=(12, 5))

    # Plot potential comparison
    plt.subplot(1, 2, 1)
    plt.plot(r_plot.flatten(), V_true.flatten(), "b-", label="True LJ Potential", linewidth=2)
    plt.plot(r_plot.flatten(), V_pred_plot.flatten(), "r--", label="NN Prediction", linewidth=2)
    plt.scatter(r_test.flatten(), V_test.flatten(), alpha=0.3, s=10, c="gray", label="Test Data")
    plt.xlabel("Distance r")
    plt.ylabel("Potential V(r)")
    plt.title("Lennard-Jones Potential Fitting")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-2, 3)

    # Plot residuals
    plt.subplot(1, 2, 2)
    residuals = V_test_pred.flatten() - V_test.flatten()
    plt.scatter(V_test.flatten(), residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("True V(r)")
    plt.ylabel("Residual (Predicted - True)")
    plt.title("Prediction Residuals")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Test model saving and loading
    print("\nTesting model save/load functionality...")
    model.save("lj_model.pkl")

    # Create a new model and load weights
    model_loaded = Sequential(Linear(1, 32), Tanh(), Linear(32, 64), ReLU(), Linear(64, 32), Tanh(), Linear(32, 1))
    model_loaded.load("lj_model.pkl")

    # Test that loaded model gives same predictions
    V_loaded_pred = model_loaded.forward(r_test)
    diff = numpy.mean(numpy.abs(V_loaded_pred - V_test_pred))
    print(f"Difference between original and loaded model: {diff:.10f}")

    print("\nTutorial completed successfully!")
    return model, optimizer


if __name__ == "__main__":
    # Run the test
    model, optimizer = test_nn_lj_fitting()

    print("\n" + "=" * 60)
    print("TUTORIAL SUMMARY")
    print("=" * 60)
    print("✓ Implemented modular neural network framework")
    print("✓ Created MSE and MAE loss functions")
    print("✓ Implemented ReLU, Sigmoid, and Tanh activations")
    print("✓ Built Linear layer with proper backpropagation")
    print("✓ Created Sequential container for model building")
    print("✓ Implemented SGD optimizer with loss visualization")
    print("✓ Successfully fitted Lennard-Jones potential")
    print("✓ Demonstrated model save/load functionality")
    print("\nStudents can now modify the architecture, try different")
    print("activation functions, loss functions, and learning rates!")
