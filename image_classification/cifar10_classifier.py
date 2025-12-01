import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.vision.transforms as transforms
from paddle.io import DataLoader
from paddle.vision.datasets import Cifar10
import numpy as np
import cv2
import os

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Define an improved Convolutional Neural Network
class CIFAR10Classifier(nn.Layer):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()
        # First conv block
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2D(32)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2D(32)
        
        # Second conv block
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2D(64)
        self.conv4 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2D(64)
        
        # Third conv block
        self.conv5 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2D(128)
        self.conv6 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2D(128)
        
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Block 1: 32x32 -> 16x16
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Block 2: 16x16 -> 8x8
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Block 3: 8x8 -> 4x4
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        
        x = paddle.flatten(x, start_axis=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Training function
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch_id, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.numpy().item()
            pred = paddle.argmax(outputs, axis=1)
            correct += (pred == labels.squeeze()).sum().numpy().item()
            total += labels.shape[0]
            
            if batch_id % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_id}], Loss: {loss.numpy().item():.4f}')
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] Complete - Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with paddle.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            pred = paddle.argmax(outputs, axis=1)
            correct += (pred == labels.squeeze()).sum().numpy().item()
            total += labels.shape[0]
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# Predict on a single custom image
def predict_image(model, image_path):
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32, 32))
    
    # Apply transforms manually (same as test transform)
    image = image.transpose((2, 0, 1)).astype('float32')  # HWC -> CHW
    # Normalize with CIFAR-10 mean and std (scaled to 0-255 range)
    mean = np.array([0.4914 * 255, 0.4822 * 255, 0.4465 * 255]).reshape(3, 1, 1)
    std = np.array([0.2470 * 255, 0.2435 * 255, 0.2616 * 255]).reshape(3, 1, 1)
    image = (image - mean) / std
    image = np.expand_dims(image, axis=0).astype('float32')  # Add batch dimension
    
    # Convert to paddle tensor
    image_tensor = paddle.to_tensor(image)
    
    # Predict
    with paddle.no_grad():
        output = model(image_tensor)
        probabilities = paddle.nn.functional.softmax(output, axis=1)
        pred_class = paddle.argmax(output, axis=1).numpy()[0]
        confidence = probabilities.numpy()[0][pred_class]
    
    print(f'Predicted: {CIFAR10_CLASSES[pred_class]} ({confidence*100:.2f}% confidence)')
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
                    'predicted_class': CIFAR10_CLASSES[pred_class],
                    'confidence': confidence
                })
    
    return results

# Save model
def save_model(model, path='cifar10_model.pdparams'):
    paddle.save(model.state_dict(), path)
    print(f'Model saved to {path}')

# Load model
def load_model(model, path='cifar10_model.pdparams'):
    model.set_state_dict(paddle.load(path))
    print(f'Model loaded from {path}')
    return model

# Custom transform class for proper normalization
class NormalizeImage:
    """Normalize image after converting to float [0,1]"""
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype='float32').reshape(3, 1, 1)
        self.std = np.array(std, dtype='float32').reshape(3, 1, 1)
    
    def __call__(self, img):
        img = img.astype('float32')
        img = (img - self.mean) / self.std
        return img

# Main function
def main():
    # CIFAR-10 normalization values
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    # Data preprocessing for training (with augmentation)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.Transpose(),  # HWC -> CHW
        transforms.Normalize(mean=[m * 255 for m in mean], std=[s * 255 for s in std])
    ])
    
    # Data preprocessing for testing (no augmentation)
    transform_test = transforms.Compose([
        transforms.Transpose(),  # HWC -> CHW
        transforms.Normalize(mean=[m * 255 for m in mean], std=[s * 255 for s in std])
    ])

    # Load CIFAR-10 dataset
    train_dataset = Cifar10(mode='train', transform=transform_train)
    test_dataset = Cifar10(mode='test', transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    # Initialize model, criterion, and optimizer
    model = CIFAR10Classifier()
    criterion = nn.CrossEntropyLoss()
    
    # Use learning rate scheduler for better training
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.001, T_max=20)
    optimizer = optim.Adam(parameters=model.parameters(), learning_rate=scheduler, weight_decay=1e-4)

    # Train the model
    train(model, train_loader, criterion, optimizer, num_epochs=20)
    
    # Evaluate on test set
    evaluate(model, test_loader)
    
    # Save the trained model
    save_model(model, 'cifar10_model.pdparams')
    
    # Example: Predict on custom image (uncomment to use)
    predict_image(model, 'frog.png')
    predict_image(model, 'Lisa.png')
    # Example: Predict on folder of images (uncomment to use)
    # results = predict_folder(model, 'path/to/your/folder')

if __name__ == '__main__':
    main()