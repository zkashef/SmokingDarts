from PIL import Image
from pillow_heif import register_heif_opener

import matplotlib.pyplot as plt

register_heif_opener()

image = Image.open('/Users/zach/repos/dartboard-cv-score-system/data/heic/IMG_8540.HEIC')

plt.imshow(image)
plt.show()