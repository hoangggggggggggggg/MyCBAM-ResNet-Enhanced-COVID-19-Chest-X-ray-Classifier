from torch.utils.data import Dataset
import os
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
import torch

class CovidDataset(Dataset):
    def __init__(self, root, is_train = "train", transform = None):
        if is_train == "train":
            root_path = os.path.join(root,"Train")
        elif is_train =="validation":
            root_path = os.path.join(root,"Val" )
        elif is_train == "test":
            root_path = os.path.join(root,"Test")
        self.categories = ['COVID-19','Normal','Non-COVID']

        self.all_image_paths = []
        self.all_labels=[]

        for index, category in enumerate(self.categories):
            sub_root_path = os.path.join(root_path, category, "images")
            for item in os.listdir(sub_root_path):
                image_path = os.path.join(sub_root_path, item)
                self.all_image_paths.append(image_path)
                self.all_labels.append(index)

        self.transform = transform
    def __len__(self):

        return len(self.all_labels)


    def __getitem__(self, item):
        item_image_path = self.all_image_paths[item]
        image = Image.open(item_image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        label = self.all_labels[item]
        return image, label

#calculate mean and std
def compute_mean_std(dataloader):
    mean = torch.zeros(1)
    std = torch.zeros(1)
    num_samples = 0

    for images, _ in dataloader:
        batch_size = images.size(0)
        num_samples += batch_size
        mean += images.mean(dim=[0, 2, 3]) * batch_size
        std += images.std(dim=[0, 2, 3]) * batch_size

    mean /= num_samples
    std /= num_samples
    return mean, std

if __name__ == '__main__':
    root_path = "C:\\Users\\laptop\\Downloads\\latest_updated_covid\\Lung Segmentation Data\\Lung Segmentation Data"

    # Temporary transform before computing mean and std
    temp_transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])

    temp_dataset = CovidDataset(root=root_path, is_train="train", transform=temp_transform)
    dataloader = DataLoader(temp_dataset, batch_size=16, shuffle=True, num_workers=0)
    for images, labels in dataloader:
        print(images.shape)
        print(labels)
    mean, std = compute_mean_std(dataloader)
    print(f"Mean: {mean.item():.4f}")
    print(f"Std: {std.item():.4f}")