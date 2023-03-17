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
"""
# Save some example images with labels
for i in range(5):
    im = Image.fromarray(np.hstack([X[i, 0], X[i, 1]]))
    ImageDraw.Draw(im).text((10, 10), y[i], font=font, fill=(255, 255, 0))
    im.save(f"./figs/image_{i}.png")
"""

class SymmetryGroup():
    def transform(self, x, B):
        """
            Returns a transformed view of x. 
            Which transforms to apply is determined by the boolean array B.

                B = [mh left, mh right, mv left, mv right, swap left and right] 

            Note: Rotations are obtained by applying both mirroring operations mh and mv.
        """
        assert len(B) == 5, "Boolean array B should be of length 5 (not {len(B)})."
        y = x.copy()
        if B[0]:
            self.mh(y, 0)
        if B[1]:
            self.mh(y, 1)
        if B[2]:
            self.mv(y, 0)
        if B[3]:
            self.mv(y, 1)
        if B[4]:
            self.s(y)
        return y

    def all_transforms(self, x):
        """
            Return all possible transformed views of x.
        """
        transformed_images = []
        # Loop over all 32 possible transforms
        for B in product(*[[0, 1]]*5):
            y = self.transform(x, B)
            transformed_images.append(np.hstack([y[0], y[1]]))
        return transformed_images

    def s(self, x):
        """
            Switch left and right image
        """
        x[[0, 1]] = x[[1, 0]]

    def mv(self, x, view=0):
        """
            Mirror along vertical axis 
        """
        x[view] = np.flip(x[view], 1)

    def mh(self, x, view=0):
        """
            Mirror along horizontal axis 
        """
        x[view] = np.flip(x[view], 0)

from itertools import product

x = X[678]
G = SymmetryGroup()

transformed_images = G.all_transforms(x) 

#im = Image.fromarray(np.vstack(transformed_images))
#im.show()

"""
im = Image.fromarray(np.vstack([x[0], G.r(x)[0], G.mv(x)[0], G.mh(x)[0]]))
im.show()

y = G.s(x)

"""
