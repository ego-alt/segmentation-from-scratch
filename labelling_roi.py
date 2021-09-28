from read_roi import read_roi_zip
from skimage.draw import polygon2mask
from skimage.measure import label, regionprops
import numpy as np

zip_file = ''


class RoiImageProcessor:
    def __init__(self, zip_file):
        self.zip_file = zip_file
        self.masks = []
        self.boxes = []

    def handle(self, size):
        for val in read_roi_zip(zip_file).values():
            coord = zip(val['x'], val['y'])
            coord = np.array([list(a) for a in coord])

            mask = polygon2mask(size, coord)
            self.masks.append(mask)
            for r in regionprops(label(mask)):
                self.boxes.append(r.bbox)
        return self.masks, self.boxes


process = RoiImageProcessor(zip_file)
process.handle((383, 512))

