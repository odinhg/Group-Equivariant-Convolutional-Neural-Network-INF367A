from utils import * 

seed_random_generators()

dl, _, _ = create_dataloaders(batch_size=8, val=0.4, test=0.4)

for x in dl:
    images, labels = x[0], x[1]
    print(images.shape)
    # Use stack to add a new dimension, cat to concat in batch dimension
    images = torch.cat([images, d2_r(images), d2_mh(images), d2_mv(images)], dim=0)
    print(images.shape)
    break
