import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class EyeTrackingModel(nn.Module):
    def __init__(self):
        super(EyeTrackingModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(16 * 25 * 12, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)  # Output layer for x and y coordinates

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class EyeDataset(Dataset):
    def __init__(self, eye_images, coords):
        self.eye_images = torch.FloatTensor(eye_images).unsqueeze(1)  # Add channel dimension
        self.coords = torch.FloatTensor(coords)

    def __len__(self):
        return len(self.eye_images)

    def __getitem__(self, idx):
        return self.eye_images[idx], self.coords[idx]

def load_and_preprocess_data(eye_images_file, mouse_coords_file):
    eye_images = np.load(eye_images_file)
    mouse_coords = np.load(mouse_coords_file)
    
    # Ensure eye_images are in the correct shape (num_samples, 100, 50)
    if eye_images.shape[1:] != (100, 50):
        eye_images = np.array([cv2.resize(img, (50, 100)) for img in eye_images])
    
    return eye_images, mouse_coords

def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def preprocess_dataset(eye_images, mouse_coords):
    processed_images = []
    for img in eye_images:
        enhanced = enhance_contrast(img)
        processed_images.append(enhanced)
    return np.array(processed_images), mouse_coords

def normalize_data(images, coords):
    image_scaler = MinMaxScaler()
    coord_scaler = MinMaxScaler()
    
    images_flat = images.reshape(images.shape[0], -1)
    normalized_images = image_scaler.fit_transform(images_flat).reshape(images.shape)
    normalized_coords = coord_scaler.fit_transform(coords)
    
    return normalized_images, normalized_coords, image_scaler, coord_scaler

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, coords in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, coords)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, coords in val_loader:
                outputs = model(images)
                loss = criterion(outputs, coords)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actual = []
    
    with torch.no_grad():
        for images, coords in test_loader:
            outputs = model(images)
            predictions.extend(outputs.numpy())
            actual.extend(coords.numpy())
    
    predictions = np.array(predictions)
    actual = np.array(actual)
    
    # Metrics
    mse = np.mean((predictions - actual) ** 2)
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)
    
    print(f"Mean Squared Error on Test Set: {mse:.4f}")
    print(f"Mean Absolute Error on Test Set: {mae:.4f}")
    print(f"R-squared on Test Set: {r2:.4f}")
    
    # Accuracy within a tolerance threshold
    tolerance = 5.0  # Adjust tolerance as needed
    within_tolerance = np.sqrt(np.sum((predictions - actual) ** 2, axis=1)) < tolerance
    accuracy = np.mean(within_tolerance) * 100
    print(f"Accuracy within {tolerance} units: {accuracy:.2f}%")
    
    return predictions, actual, mse, mae, r2, accuracy

def plot_results(train_losses, val_losses, predictions, actual):
    plt.figure(figsize=(16, 10))
    
    # Loss Plot
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Actual vs Predicted Scatter Plot
    plt.subplot(2, 2, 2)
    plt.scatter(actual[:, 0], actual[:, 1], c='blue', label='Actual', alpha=0.5)
    plt.scatter(predictions[:, 0], predictions[:, 1], c='red', label='Predicted', alpha=0.5)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.title('Actual vs Predicted Coordinates')
    
    # Error Distribution
    plt.subplot(2, 2, 3)
    errors = np.sqrt(np.sum((predictions - actual) ** 2, axis=1))
    plt.hist(errors, bins=20, color='purple', alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution')
    
    # Convergence Plot of MSE
    plt.subplot(2, 2, 4)
    mse_train_val_diff = np.abs(np.array(train_losses) - np.array(val_losses))
    plt.plot(mse_train_val_diff, label="Train-Validation MSE Difference", color="orange")
    plt.xlabel('Epoch')
    plt.ylabel('MSE Difference')
    plt.legend()
    plt.title('Convergence Rate (Train-Validation MSE Difference)')
    
    plt.tight_layout()
    plt.show()
from torchviz import make_dot

def plot_model_architecture(model):
    # Generate a dummy input with the same shape as your input data
    dummy_input = torch.randn(1, 1, 100, 50)  # Batch size 1, 1 channel, height 100, width 50
    model_architecture = make_dot(model(dummy_input), params=dict(model.named_parameters()))
    
    # Display the model architecture plot
    model_architecture.render("model_architecture", format="png")
    print("Model architecture saved as 'model_architecture.png'")


def print_model(model):
    """
    Prints a summary of the model architecture.
    
    Parameters:
    model (torch.nn.Module): The PyTorch model to be printed.
    """
    print(model)
    print("\nModel Summary:")
    
    # Get the summary of the model
    from torchsummary import summary
    summary(model, input_size=(1, 224, 224))
def main():
    # Load and preprocess data
    eye_images, mouse_coords = load_and_preprocess_data(
        r"C:\Users\aryaa\Documents\PicsCOLAB\eyes\eye_images.npy",
        r"C:\Users\aryaa\Documents\PicsCOLAB\eyes\mouse_coords.npy"
    )
    
    # Preprocess the dataset
    processed_images, valid_mouse_coords = preprocess_dataset(eye_images, mouse_coords)
    
    # Normalize the data
    norm_images, norm_coords, image_scaler, coord_scaler = normalize_data(processed_images, valid_mouse_coords)
    
    # Split the data into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(norm_images, norm_coords, test_size=0.2, random_state=49)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=49)
    
    # Create datasets and dataloaders
    train_dataset = EyeDataset(X_train, y_train)
    val_dataset = EyeDataset(X_val, y_val)
    test_dataset = EyeDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize and train the model
    model = EyeTrackingModel()
    
    # Plot the model architecture
    
    train_losses, val_losses = train_model(model, train_loader, val_loader)
    
    # Evaluate the model with additional metrics
    predictions, actual, mse, mae, r2, accuracy = evaluate_model(model, test_loader)
    print_model(model)
    # Plot results with additional insights
    plot_results(train_losses, val_losses, predictions, actual)
    
    # Save the model
    torch.save(model.state_dict(), 'eye_tracking_model.pth')
    
    print("Model training and evaluation complete. Model saved as 'eye_tracking_model.pth'")

if __name__ == "__main__":
    main()
