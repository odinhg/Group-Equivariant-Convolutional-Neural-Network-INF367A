import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.functional import to_tensor, normalize
from os.path import isfile
from PIL import Image

def seed_random_generators(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def merge_views(x: np.ndarray) -> np.ndarray:
    """
        Take a RGB stereo image of shape (2, H, W, 3) as input and returns a merged version of shape (H, 2*W, 3).
    """
    y = np.hstack([x[0], x[1]]) 
    return y

def split_views(y: np.ndarray) -> np.ndarray:
    """
        Take a merged stereo image of shape (H, 2*W, 3) as input and returns a expanded version of shape (2, H, W, 3).
    """

    c = y.shape[1] // 2
    x = np.array([y[:, 0:c, :], y[:, c:, :]])
    return x

def numpy_image_to_tensor(x: np.ndarray) -> torch.FloatTensor:
    """
        Convert numpy image with values 0 - 255 to torch tensor with values 0 - 1.
    """
    return torch.from_numpy(x / 255).type(torch.FloatTensor)

def tensor_to_numpy_image(x: torch.FloatTensor) -> np.ndarray:
    """
        Convert tensor image with values 0 - 1 to numpy image with values 0 - 255.
    """
    return (x.cpu().detach() * 255).type(torch.uint8).numpy()

def visualize_tensor(x: torch.FloatTensor, filename: str = None) -> None:
    """
        Take tensor image and display left and right view side-by-side. If filename is provided, save image instead of displaying it.
    """
    y = tensor_to_numpy_image(x)
    image = Image.fromarray(y)
    if filename:
        image.save(filename)
    else:
        image.show()

class ImageDataset(Dataset):
    """ 
        Dataset class for stereo images. 
    """
    def __init__(self, images_file: str = "data/X.npy", labels_file: str = "data/y.npy") -> None:
        
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

        if len(self.images) != len(self.labels):
            raise ValueError(f"Error: Number of images and labels not equal.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.FloatTensor, torch.uint8]:
        image = self.images[idx]
        label = self.labels[idx]
        image = merge_views(image)
        image = numpy_image_to_tensor(image) 
        label = torch.tensor(label, dtype=torch.uint8)
        return (image, label)

    def label_str(self, label: int) -> str:
        return self.label_list[label]

    def mean(self) -> np.ndarray:
        self.mean = np.mean(self.images, axis=0)
        return self.mean 

    def std(self) -> np.ndarray:
        self.std = np.std(self.images, axis=0)
        return self.std

def create_dataloaders(batch_size: int, test: float, val: float, random_seed: int = 0) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
        Create data loaders for training, validation and test datasets.
    """
    dataset = ImageDataset()
    generator = torch.Generator().manual_seed(random_seed)

    val_size = int(val * len(dataset))
    test_size = int(test * len(dataset))
    train_size = len(dataset) - val_size - test_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return (train_dl, val_dl, test_dl)

def d2_mh(x: torch.FloatTensor) -> torch.Tensor:
    """
        Mirror image tensor around horizontal axis. Supports mini-batches.
    """
    if len(x.shape) > 3:
        return torch.flip(x, [1])
    return torch.flip(x, [0])

def d2_mv(x: torch.FloatTensor) -> torch.Tensor:
    """
        Mirror image tensor around vertical axis. Supports mini-batches.
    """
    if len(x.shape) > 3:
        return torch.flip(x, [2])
    return torch.flip(x, [1])

def d2_r(x: torch.FloatTensor) -> torch.Tensor:
    """
        Rotate image tensor 180 degrees CCW around center. Supports mini-batches.
    """
    if len(x.shape) > 3:
        return torch.flip(x, [1, 2])
    return torch.flip(x, [0, 1])
