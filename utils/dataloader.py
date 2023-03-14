import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, normalize
from os.path import isfile

class ImageDataset(Dataset):
    """ Custom dataset for stereo images """
    def __init__(self, images_file="data/X.npy", labels_file="data/y.npy"):
        assert isfile(images_file), f"Error: Images file {images_file} not found!"
        assert isfile(labels_file), f"Error: Labels file {labels_file} not found!"
        self.images_file = images_file
        self.labels_file = labels_file
        self.images = np.load(self.images_file)
        self.labels = np.load(self.labels_file)
        self.label_list, self.labels = np.unique(self.labels, return_inverse=True)
        assert len(self.images) == len(self.labels), f"Error: Number of images and labels not equal!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return (image, label)

    def label_str(self, label):
        return self.label_list[label]

    def mean(self):
        self.mean = np.mean(self.images, axis=0)
        return self.mean 

    def std(self):
        self.std = np.std(self.images, axis=0)
        return self.std

