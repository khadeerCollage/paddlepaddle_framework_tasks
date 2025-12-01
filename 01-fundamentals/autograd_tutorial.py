import paddle

# Example of a simple tensor operation
x = paddle.to_tensor([1.0, 2.0, 3.0])
y = paddle.to_tensor([4.0, 5.0, 6.0])

# Basic arithmetic operations
sum_result = paddle.add(x, y)
print("Sum:", sum_result.numpy())

# Automatic differentiation
x.stop_gradient = False  # Enable gradient tracking
y = x ** 2  # Define a tensor operation
y.backward()  # Compute gradients

# Display gradients
print("Gradient of x:", x.grad.numpy())