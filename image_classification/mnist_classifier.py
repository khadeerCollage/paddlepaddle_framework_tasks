import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import DataLoader
import paddle.vision.transforms as transforms
from paddle.vision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Define a CNN for better accuracy (instead of simple feedforward)
class MNISTClassifier(nn.Layer):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2D(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2D(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2D(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2D(32)
        self.bn2 = nn.BatchNorm2D(64)
        self.bn3 = nn.BatchNorm2D(128)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1: 28x28 -> 14x14
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        # Conv block 2: 14x14 -> 7x7
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        # Conv block 3: 7x7 -> 3x3
        x = self.pool(nn.functional.relu(self.bn3(self.conv3(x))))
        # Flatten and FC
        x = paddle.flatten(x, start_axis=1)
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Simple feedforward neural network (alternative, lower accuracy)
class SimpleNN(nn.Layer):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.reshape([-1, 28 * 28])  # Flatten the input
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training loop
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch_id, (data, label) in enumerate(train_loader):
            output = model(data)
            loss = criterion(output, label)
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.numpy().item()
            pred = paddle.argmax(output, axis=1)
            correct += (pred == label.squeeze()).sum().numpy().item()
            total += label.shape[0]

            if batch_id % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_id}], Loss: {loss.numpy().item():.4f}')
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}] Complete - Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with paddle.no_grad():
        for data, label in test_loader:
            output = model(data)
            pred = paddle.argmax(output, axis=1)
            total += label.shape[0]
            correct += (pred == label.squeeze()).sum().numpy().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# Predict on a single custom image
def predict_image(model, image_path):
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None
    
    # Resize to 28x28
    image = cv2.resize(image, (28, 28))
    
    # Invert if background is white (MNIST has black background)
    if np.mean(image) > 127:
        image = 255 - image
    
    # Normalize same as training (MNIST mean/std)
    image = image.astype('float32') / 255.0
    image = (image - 0.1307) / 0.3081
    
    # Add batch and channel dimensions: (1, 1, 28, 28)
    image = np.expand_dims(image, axis=(0, 1))
    
    # Convert to paddle tensor
    image_tensor = paddle.to_tensor(image)
    
    # Predict
    with paddle.no_grad():
        output = model(image_tensor)
        probabilities = paddle.nn.functional.softmax(output, axis=1)
        pred_class = paddle.argmax(output, axis=1).numpy()[0]
        confidence = probabilities.numpy()[0][pred_class]
    
    print(f'Predicted Digit: {pred_class} ({confidence*100:.2f}% confidence)')
    return pred_class, confidence

# Predict on multiple images in a folder
def predict_folder(model, folder_path):
    model.eval()
    results = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            pred_class, confidence = predict_image(model, image_path)
            if pred_class is not None:
                results.append({
                    'filename': filename,
                    'predicted_digit': pred_class,
                    'confidence': confidence
                })
    
    return results

# Visualize predictions on test samples
def visualize_predictions(model, test_loader, num_samples=10):
    model.eval()
    images, labels = next(iter(test_loader))
    
    with paddle.no_grad():
        outputs = model(images[:num_samples])
        predictions = paddle.argmax(outputs, axis=1).numpy()
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        img = images[i].numpy().squeeze()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Pred: {predictions[i]}, True: {labels[i].numpy()[0]}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('mnist_predictions.png')
    plt.show()
    print("Visualization saved to mnist_predictions.png")

# Save model
def save_model(model, path='mnist_model.pdparams'):
    paddle.save(model.state_dict(), path)
    print(f'Model saved to {path}')

# Load model
def load_model(model, path='mnist_model.pdparams'):
    model.set_state_dict(paddle.load(path))
    print(f'Model loaded from {path}')
    return model

# Main function
def main():
    # Data transforms (simpler without RandomRotation to avoid PIL compatibility issues)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])  # MNIST mean/std
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])

    # Load the MNIST dataset
    train_dataset = MNIST(mode='train', transform=train_transform)
    test_dataset = MNIST(mode='test', transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize the CNN model (better accuracy than SimpleNN)
    model = MNISTClassifier()
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr.StepDecay(learning_rate=0.001, step_size=3, gamma=0.5)
    optimizer = optim.Adam(parameters=model.parameters(), learning_rate=scheduler)

    # Train the model
    train(model, train_loader, criterion, optimizer, epochs=10)
    
    # Evaluate on test set
    evaluate(model, test_loader)
    
    # Save the trained model
    save_model(model, 'mnist_model.pdparams')
    
    # Visualize some predictions
    visualize_predictions(model, test_loader)
    
    # Example: Predict on custom image (uncomment to use)
    # predict_image(model, 'path/to/your/digit.jpg')
    
    # Example: Predict on folder of images (uncomment to use)
    # results = predict_folder(model, 'path/to/your/folder')

if __name__ == '__main__':
    main()