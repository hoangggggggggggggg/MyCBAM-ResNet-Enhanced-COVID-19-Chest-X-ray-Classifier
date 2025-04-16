import torch
from Covid_dataset_and_calculate_mean_and_std import CovidDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.autonotebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Covid_model_based_residual_network_and_CBAM import Model
from sklearn.metrics import classification_report

# Function to plot the normalized confusion matrix as a heatmap
def plot_confusion_matrix(cm, class_names, epoch):
    fig, ax = plt.subplots()

    # Normalize the confusion matrix by row
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a heatmap using seaborn
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=class_names, yticklabels=class_names)

    # Set labels and title
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(f"Confusion Matrix - Epoch {epoch}")
    # Adjust layout to prevent label cut-offs
    plt.tight_layout()
    plt.show()

# Function to evaluate the model on the test dataset
def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define preprocessing pipeline for input images
    transform = Compose([
        ToTensor(),
        Resize((224, 224)),
        Normalize(mean=[0.5143], std=[0.2461])
    ])

    # Load test dataset using custom CovidDataset class
    test_dataset = CovidDataset(root="C:\\Users\\laptop\\Downloads\\latest_updated_covid\\Lung Segmentation Data\\Lung Segmentation Data",
                                is_train="test", transform=transform)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True, num_workers=4, drop_last=False)

    # Initialize the model and load pretrained weights
    model = Model()
    checkpoint = torch.load("C:\\Users\\laptop\\PycharmProjects\\PythonProject2\\train_models\\New_Covid_Model_with_CBAM_Adam\\best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Lists to store predictions and labels for evaluation
    all_predictions = []
    all_labels = []

    # Test mode
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(test_dataloader, colour="blue")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            predictions = torch.argmax(predictions, 1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    # Calculate amatrix
    acc = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    # Print results
    print(f"Accuracy: {acc:.4f}")
    print(report)
    # Plot confusion matrix
    class_names = ['COVID-19','Normal','Non-COVID']
    plot_confusion_matrix(cm, class_names, epoch=1)


if __name__ == '__main__':
    test()
