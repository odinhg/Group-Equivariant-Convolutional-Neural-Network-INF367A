import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, normalize
from os.path import isfile
from PIL import Image

def merge_views(x: np.ndarray) -> np.ndarray:
    """
        Take a RGB stereo image of shape (2, H, W, 3) as input and returns a merged version of shape (W, H, 6) by stacking channels in the last dimension.
    """

    if (x.shape[0] != 2) or (x.shape[3] != 3):
        raise ValueError(f"Unexpected dimensions. Expected first and last dimensions of size 2 and 3. Got {x.shape[0]} and {x.shape[3]}.")

    y = np.swapaxes(x, 0, 2)
    y = y.reshape(y.shape[0], y.shape[1], 6)
    return y

def split_views(y: np.ndarray) -> np.ndarray:
    """
        Take a merged stereo image of shape (W, H, 6) as input and returns a expanded version of shape (2, H, W, 3). This is the inverse of merge_views().
    """
    
    if y.shape[2] != 6:
        raise ValueError(f"Unexpected dimensions. Expected last dimension of size 6. Got {y.shape[2]}.")

    x = y.reshape(y.shape[0], y.shape[1], 2, 3)
    x = np.swapaxes(x, 0, 2)
    return x

def d2_mh(y: np.ndarray) -> np.ndarray:
    """
        Takes merged stereo image of shape (W, H, 6). Returns the image flipped around the horizontal axis.
    """

    y = np.flip(y, 1)
    return y

def d2_mv(y: np.ndarray) -> np.ndarray:
    """
        Takes a merged stereo image of shape (W, H, 6). Returns the image flipped around the vertical axis.
    """
    y = np.flip(y, 0)
    y = np.roll(y, 3, axis=2)
    return y

def d2_r(y: np.ndarray) -> np.ndarray:
    """
        Takes a merged stereo image of shape (W, H, 6). Returns a rotated version of the image.
    """
    y = np.flip(y, 0)
    y = np.flip(y, 1)
    y = np.roll(y, 3, axis=2)
    return y

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
    y = split_views(y)
    image = Image.fromarray(np.hstack([y[0], y[1]]))
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


"""

TODO:
    - Create dataloader function to return train, val and test data loaders.
        - Should take val size, test size, batch size as arguments.
    - Change symmetries to work with tensors instead of numpy arrays.
        - NB! Should work with mini-batches of shape (B, W, H, 6).
    - Create some example images showing the different symmetries and add them to the report.
"""
