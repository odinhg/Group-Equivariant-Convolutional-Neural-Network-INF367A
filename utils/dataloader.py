import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Resize
from os.path import isfile

from .misc import numpy_image_to_tensor

class ImageDataset(Dataset):
    """ 
        Dataset class for stereo images. 
    """
    def __init__(self, images_file: str = "data/X.npy", labels_file: str = "data/y.npy", 
                 image_size: tuple[int, int] = (200, 400)) -> None:
        
        if not isfile(images_file):
            raise ValueError(f"Error: Images file {images_file} does not exist.")
        
        if not isfile(labels_file):
            raise ValueError(f"Error: Labels file {labels_file} does not exist.")
        
        self.images_file = images_file
        self.labels_file = labels_file
        
        # Load images and labels
        self.images = np.load(self.images_file)
        self.labels = np.load(self.labels_file)

        # Create lists containing label strings. In our case, ["cloudy", "sunny"] and [0, 1].
        self.label_list, self.labels = np.unique(self.labels, return_inverse=True)

        self.resize_transform = Resize(size=image_size, antialias=False)

        if len(self.images) != len(self.labels):
            raise ValueError(f"Error: Number of images and labels not equal.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.float32]:
        """
            Returns a stereo image of shape (3, 2, H, W).
        """
        image = self.images[idx]
        label = self.labels[idx]
        image = numpy_image_to_tensor(image) 
        image = self.resize_transform(image)
        label = torch.tensor(label, dtype=torch.float32)
        return (image, label)

    def label_str(self, label: int) -> str:
        return self.label_list[label]

def seed_worker(worker_id):
    """
        From PyTorch docs. Ensures deterministic dataloader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloaders(batch_size: int, test: float, val: float, image_size: tuple[int, int] = (200, 400), 
                        random_seed: int = 0, num_workers: int = 8) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
        Create data loaders for training, validation and test datasets.
    """
    dataset = ImageDataset(image_size=image_size)
    generator = torch.Generator().manual_seed(random_seed)

    val_size = int(val * len(dataset))
    test_size = int(test * len(dataset))
    train_size = len(dataset) - val_size - test_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, num_workers=num_workers)

    return (train_dl, val_dl, test_dl)

