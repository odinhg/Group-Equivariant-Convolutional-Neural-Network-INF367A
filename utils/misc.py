import numpy as np
import torch
from PIL import Image
import random

def seed_random_generators(seed: int = 0, deterministic: bool = True) -> None:
    """
        Seed random generators with given seed. 
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

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
        Output tensor of shape (3, H, W).
    """
    y = torch.from_numpy(x / 255).type(torch.FloatTensor)
    y = torch.permute(y, (2, 0, 1))
    return y


def tensor_to_numpy_image(x: torch.FloatTensor) -> np.ndarray:
    """
        Convert tensor image with values 0 - 1 to numpy image with values 0 - 255.
        Output numpy array of shape (H, W, 3)
    """
    y = torch.permute(x, (1, 2, 0))
    return (y.cpu().detach() * 255).type(torch.uint8).numpy()

def visualize_tensor(x: torch.FloatTensor, filename: str = "") -> None:
    """
        Take tensor image and display left and right view side-by-side. If filename is provided, save image instead of displaying it.
    """
    y = tensor_to_numpy_image(x)
    image = Image.fromarray(y)
    if filename:
        image.save(filename)
    else:
        image.show()

def d2_mh(x: torch.Tensor) -> torch.Tensor:
    """
        Mirror image tensor around horizontal axis. Supports mini-batches.
    """
    return torch.flip(x, dims=[-2])

def d2_mv(x: torch.Tensor) -> torch.Tensor:
    """
        Mirror image tensor around vertical axis. Supports mini-batches.
    """
    return torch.flip(x, dims=[-1])

def d2_r(x: torch.Tensor) -> torch.Tensor:
    """
        Rotate image tensor 180 degrees CCW around center. Supports mini-batches.
    """
    return torch.flip(x, dims=[-1, -2])

def d2_e(x: torch.Tensor) -> torch.Tensor:
    """
        Identity.
    """
    return x
