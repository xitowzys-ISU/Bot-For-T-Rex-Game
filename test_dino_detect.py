import cv2
import numpy as np
from mss import mss
from PIL import Image
import matplotlib.pyplot as plt

from skimage import io
from skimage import data
from skimage.feature import match_template


class Object:
    def __init__(self, path):
        self.img = io.imread(path)
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        self.location = None


player = Object('./objects/dino_day.png')
scr = np.invert(io.imread('./dino_game.png'))

result = match_template(scr, np.invert(player.img))

ij = np.unravel_index(np.argmax(result), result.shape)
# print(ij)
print(ij[::-1])
_, x, y = ij[::-1]

hcoin, wcoin, _ = player.img.shape
print(player.img.shape)

fig, ax = plt.subplots()

ax.imshow(scr)
rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='g', facecolor='none')
ax.add_patch(rect)

plt.show()
