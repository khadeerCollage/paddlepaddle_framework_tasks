import paddle

# Create a tensor from a list
tensor_a = paddle.to_tensor([1, 2, 3, 4, 5])
print("Tensor A:", tensor_a)

# Create a 2D tensor (matrix)
tensor_b = paddle.to_tensor([[1, 2], [3, 4]])
print("Tensor B:\n", tensor_b)

# Tensor operations
tensor_sum = tensor_a + 10
print("Tensor A + 10:", tensor_sum)

tensor_product = tensor_a * 2
print("Tensor A * 2:", tensor_product)

# Indexing
print("First element of Tensor A:", tensor_a[0].numpy())
print("Slice of Tensor A (first three elements):", tensor_a[:3].numpy())

# Reshaping
tensor_c = tensor_b.reshape([4, 1])
print("Reshaped Tensor B:\n", tensor_c)

# Transposing
tensor_d = tensor_b.transpose([1, 0])
print("Transposed Tensor B:\n", tensor_d)

# Basic statistics
mean_a = paddle.mean(tensor_a)
print("Mean of Tensor A:", mean_a.numpy())

std_a = paddle.std(tensor_a)
print("Standard deviation of Tensor A:", std_a.numpy())