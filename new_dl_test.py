import torch

from utils import create_dataloaders
from utils import visualize_tensor
from utils import d2_r, d2_e, d2_mh, d2_mv

dl, _, _ = create_dataloaders(batch_size=16, test=0.3, val=0.3, image_size = (200, 400), num_workers=2)

for data in dl:
    images, labels = data
    break

print(images.shape)

transf = [images[0], d2_r(images[0]), d2_mh(images[0]), d2_mv(images[0])]

visualize_tensor(torch.cat(transf, dim=-2))

