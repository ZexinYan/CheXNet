import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, file_list, transforms=None):
        file_names = []
        labels = []
        with open(file_list, 'r') as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                file_names.append(image_name)
                labels.append(label)

        self.file_names = file_names
        self.labels = labels
        self.transform = transforms

    def __getitem__(self, index):
        image_name = self.file_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.file_names)
