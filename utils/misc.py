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
