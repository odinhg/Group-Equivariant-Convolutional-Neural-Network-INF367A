from utils import create_dataloaders
from utils import visualize_tensor

dl, _, _ = create_dataloaders(batch_size=16, test=0.3, val=0.3, image_size = (200, 400), num_workers=2)

for data in dl:
    images, labels = data
    break

print(images.shape)

visualize_tensor(images[0])
