import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image

class customDataset(Dataset):
    def __init__(self, annotation_file, transform=None, target_transform=None):

        self.img_labels = pd.read_csv(annotation_file)

        self.transform = transform

        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        # Get image path
        img_path = self.img_labels.iloc[idx]["file name"]

        # Load PNG image
        image = decode_image(str(img_path), mode="RGB")

        # Get numerical class ID
        label = int(self.img_labels.iloc[idx]["class ID"])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
