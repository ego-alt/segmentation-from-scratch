import numpy as np
import cv2


class Resize_Crop(object):
    """"Simultaneous random transforms on images and their labels"""
    def __init__(self, crop):
        self.h, self.w = crop  # Standard cropping dimensions

    def __call__(self, image, label):
        # Random resize while preserving the aspect ratio
        height, width = image.shape[0:2]
        scale = np.random.randint(50, 300) / 100  # 0.5x to 3x
        dim = (int(width * scale), int(height * scale))
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, dim, interpolation=cv2.INTER_NEAREST)

        if len(label.shape) == 2:
            label = np.expand_dims(label, axis=-1)
        height, width, num = label.shape  # New dimensions
        while True:  # Random crop to the required dimensions
            y = np.random.randint(0, height - self.h)
            x = np.random.randint(0, width - self.w)
            condition = label[y:y + self.h, x:x + self.w, :]
            if len(np.unique(condition[:, :, num - 1])) > 1:
                label = condition  # Processed label
                image = image[y:y + self.h, x:x + self.w]  # Processed image
                break

        return image, label
