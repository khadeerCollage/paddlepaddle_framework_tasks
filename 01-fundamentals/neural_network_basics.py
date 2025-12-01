import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class SimpleNeuralNetwork(nn.Layer):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input layer to hidden layer
        self.fc2 = nn.Linear(128, 10)    # Hidden layer to output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function
        x = self.fc2(x)          # Output layer
        return x

def main():
    # Example usage
    model = SimpleNeuralNetwork()
    # Create a dummy input tensor with shape (batch_size, input_size)
    input_tensor = paddle.randn([32, 784])  # Batch size of 32
    output = model(input_tensor)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()