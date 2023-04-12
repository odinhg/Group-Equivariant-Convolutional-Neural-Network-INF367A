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

def numpy_image_to_tensor(x):
    """
        Convert numpy image of shape (2, H, W, 3) with values 0 - 255 to torch tensor with values 0 - 1.
        Output tensor of shape (3, 2, H, W).
    """
    y = torch.from_numpy(x / 255).type(torch.Tensor)
    y = torch.permute(y, (3, 0, 1, 2))
    return y

def tensor_to_numpy_image(x: torch.FloatTensor) -> np.ndarray:
    """
        Convert tensor image of shape (3, 2, H, W) with values 0 - 1 to numpy image with values 0 - 255.
        Output numpy array of shape (2, H, W, 3).
    """
    y = torch.permute(x, (1, 2, 3, 0))
    return (y.cpu().detach() * 255).type(torch.uint8).numpy()

def visualize_tensor(x: torch.FloatTensor, filename: str = "") -> None:
    """
        Take tensor image and display left and right view side-by-side. If filename is provided, save image instead of displaying it.
    """
    y = tensor_to_numpy_image(x)
    y = np.hstack([y[0], y[1]])
    image = Image.fromarray(y)
    if filename:
        image.save(filename)
    else:
        image.show()

def choose_model(configs):
    """ 
    Ask user to choose model/config to use.
    """
    n = -1
    while n < 0 or n >= len(configs):
        print("Select model:")
        for i, config in enumerate(configs):
            print(f"[{i}] {config['name']}")
        n = int(input() or 0)

    config = configs[n]
    print("Loaded model configuration:")
    for key, value in config.items():
        print(f"\t* {key}: {value}")

    return config
