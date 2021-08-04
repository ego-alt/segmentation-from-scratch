import numpy as np
from PIL import Image
from os import listdir
from os.path import join
import cv2


class Cluster:
    def __init__(self, array):
        cells = np.where(array > 0.5)
        full_x, full_y = cells

        self.cells = set(zip(full_x, full_y))
        self.c_num = len(self.cells)
        self.instance = np.zeros_like(array)
        self.cell_clusters = []

    def main(self, filename):
        print(f'Preparing {filename}')
        for (x, y) in self.cells:
            cluster = set()
            cluster = self.find_cell(x, y, cluster)  # Find surrounding neighbours
            if cluster: self.cell_clusters.append(cluster)
        print(f'Instances have been resolved for {filename}')

        self.save_im(filename)

    def find_neigh(self, col, row):
        x1 = (col - 1) if col > 0 else col
        x2 = (col + 1) if col < len(self.cells) else col
        y1 = (row - 1) if row > 0 else row
        y2 = (row + 1) if row < len(self.cells) else row
        neigh = [(c, r) for c in range(x1, x2 + 1) for r in range(y1, y2 + 1)]
        return neigh

    def find_cell(self, col, row, cluster):
        """Neighbours: surrounding 8 pixels & myself
        Friends: neighbouring cells"""
        friends = (self.cells & set(self.find_neigh(col, row)))
        self.cells = self.cells.difference(friends)  # Remove neighbours from consideration

        if friends:
            cluster.update(list(friends))
            for (c, r) in friends:
                self.find_cell(c, r, cluster)
        return cluster

    def clustering(self):
        """Checks whether instances of unusually small size should be merged"""
        n = self.c_num // len(self.cell_clusters)  # Average no. of pixels per cell

        for out in self.cell_clusters:
            if len(out) < (n // 10):
                x1, y1 = sorted(out)[0]
                for rem in self.cell_clusters:
                    if out != rem and set(self.find_neigh(x1, y1)) & rem:
                        self.cell_clusters.remove(out)
                        rem.update(out)

    def save_im(self, filename):
        for idx, lst in enumerate(self.cell_clusters):
            for (x, y) in lst:
                self.instance[x - 1, y - 1] = idx

        print(f'Saving {filename} ...')
        output = Image.fromarray(np.uint8(self.instance))
        output.save(filename)


class ImageProcessor:
    def __init__(self, img_path, save_path):
        self.names = [lb for lb in sorted(listdir(img_path))][0:25]
        self.labels = np.array([cv2.imread(join(img_path, lb), 0) for lb in self.names]) / 255
        self.save_path = save_path

    def main(self, crop):
        for idx, i in enumerate(self.labels):
            img = self.center_crop(i, crop, crop)
            handler = Cluster(img)
            handler.main(join(self.save_path, self.names[idx]))

    def center_crop(self, img, crop_x, crop_y):
        y, x = img.shape
        x0 = (x - crop_x) // 2
        y0 = (y - crop_y) // 2
        crop_img = img[y0:y0 + crop_y, x0:x0 + crop_x]
        return crop_img


root = " "
save_root = " "
ImageProcessor(root, save_root).main(256)
