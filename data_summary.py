import numpy as np
from PIL import Image, ImageDraw, ImageFont

X = np.load("./data/X.npy")
y = np.load("./data/y.npy")

no_images, views, height, width, channels = X.shape

print(f"Number of images: {no_images}")
print(f"Views per image: {views}")
print(f"Resolution (w x h): {width} x {height}")
print(f"Channels: {channels}")

unique, counts = np.unique(y, return_counts=True)

print(f"Unique labels: {unique}")
print(f"Label counts: {counts}")

font = ImageFont.truetype("./data/font.ttf", 32)

# Save some example images with labels
for i in range(5):
    im = Image.fromarray(np.hstack([X[i, 0], X[i, 1]]))
    ImageDraw.Draw(im).text((10, 10), y[i], font=font, fill=(255, 255, 0))
    im.save(f"./figs/image_{i}.png")
