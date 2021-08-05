import numpy as np
from PIL import Image
from os import listdir
from os.path import join
import cv2
import re


class Cluster:
    def __init__(self, array):
        cells = np.where(array == 1)
        full_x, full_y = cells

        self.cells = set(zip(full_x, full_y))
        self.c_num = len(self.cells)
        self.cell_clusters = []

    def main(self, filename):
        print(f'Preparing {filename}')
        for (x, y) in self.cells:
            cluster = set()
            cluster = self.find_cell(x, y, cluster)  # Find surrounding neighbours
            if cluster: self.cell_clusters.append(cluster)
        print(f'Instances have been resolved for {filename}')
        return self.cell_clusters

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

    """def clustering(self):
        Checks whether instances of unusually small size should be merged
        n = self.c_num // len(self.cell_clusters)  # Average no. of pixels per cell

        for out in self.cell_clusters:
            if len(out) < (n // 10):
                x1, y1 = sorted(out)[0]
                for rem in self.cell_clusters:
                    if out != rem and set(self.find_neigh(x1, y1)) & rem:
                        self.cell_clusters.remove(out)
                        rem.update(out)"""


class Batches:
    def __init__(self, names):
        self.names = names
        self.files = {}
        self.org_files()

    def org_files(self):
        regex = "^F0[1-4]_[0-9]+"
        for file in self.names:
            if not file.startswith('.'):
                filename = re.findall(regex, file)[0]
                self.listdict(self.files, filename, file)

    def listdict(self, dictionary, key, value):
        if key not in dictionary:
            dictionary[key] = list()
        dictionary[key].append(value)


class ImageProcessor:
    def __init__(self, img_path, save_path):
        self.files = Batches(sorted(listdir(img_path))).files

        self.labels = []
        for name in self.files:
            label = [cv2.imread(join(img_path, i), 0) for i in self.files[name]]
            self.labels.append(np.array(label) / 255)

        self.save_path = save_path

    def main(self, crop):
        names = [n for n in self.files]
        for idx, images in enumerate(self.labels):
            img = [self.center_crop(i, crop, crop) for i in images]
            layer_num = len(img)

            holder = []
            black_strip = np.zeros((1, crop))
            for i in img:
                holder.extend([i, black_strip])
            img = np.concatenate(holder)

            cells = Cluster(img).main(names[idx])
            instance = np.zeros_like(img)
            self.save_im(names[idx], cells, instance, layer_num)

    def center_crop(self, img, crop_x, crop_y):
        y, x = img.shape
        x0 = (x - crop_x) // 2
        y0 = (y - crop_y) // 2
        crop_img = img[y0:y0 + crop_y, x0:x0 + crop_x]
        return crop_img

    def save_im(self, name, cells, instance, layer_num):
        for idx, lst in enumerate(cells):
            for (x, y) in lst:
                instance[x, y] = idx + idx*10

        print(f'Saving {name} ...')
        inst = np.array_split(instance, layer_num)
        for ind, arr in enumerate(inst):
            output = Image.fromarray(np.uint8(arr[:-1, :]))
            filename = join(self.save_path, self.files[name][ind])
            print(filename)
            print(np.unique(arr[:-1, :]))
            output.save(filename)


root = '/Users/batfolder/Downloads/Summer project/MP6843_seg'
save = '/Users/batfolder/Downloads/Summer project/test2'
testtet = ImageProcessor(root, save)
testtet.main(256)
