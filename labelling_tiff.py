import numpy as np
from PIL import Image
from os import listdir
from os.path import join
from skimage.measure import label
import cv2
import re


"""Specifically tailored for Project P2043 of the Cell Image Library
Requirements: convert float values to binary 0/1 and label each cell instance with a new value"""


class Cluster:
    # Crude flood-fill function which was replaced by sci-kit's method
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


class Batches:
    def __init__(self, names):
        self.names = names
        self.files = {}
        self.org_files()

    def org_files(self):
        # Organises files according to the specific sequence F0X_XXXX in their filename
        regex = "^F0[1-4]_[0-9]+"
        for file in self.names:
            if not file.startswith('.'):
                filename = re.findall(regex, file)[0]
                self.listdict(self.files, filename, file)

    def listdict(self, dictionary, key, value):
        # Groups files in lists within a dictionary
        if key not in dictionary:
            dictionary[key] = list()
        dictionary[key].append(value)


class TiffImageProcessor:
    def __init__(self, img_path, save_path):
        self.files = Batches(sorted(listdir(img_path))).files
        self.labels = []
        for name in self.files:
            label = [cv2.imread(join(img_path, i), 0) for i in self.files[name]]
            # Converts float values to binary values
            l_arr = (np.array(label) / 255)
            l_arr[l_arr < 0.5] = 0
            l_arr[l_arr > 0.5] = 1
            self.labels.append(l_arr)
        self.save_path = save_path

    def handle(self, crop=None, cluster=None):
        names = [n for n in self.files]
        batch_arr = []
        layer_num = []

        for idx, img in enumerate(self.labels):
            holder = []
            if crop: img = [self.center_crop(i, crop, crop) for i in img]
            layer_num.append(len(img))
            for i in img:
                _, x = i.shape
                black_strip = np.zeros((1, x))
                holder.extend([i, black_strip])
            img = np.concatenate(holder)
            if cluster: self.cluster(img, names[idx], len(img))

            batch_arr.append(img)

        return names, batch_arr, layer_num

    def cluster(self, img, filename, layer_num):
        cells = Cluster(img).main(filename)
        instance = np.zeros_like(img)
        for i, lst in enumerate(cells):
            for (x, y) in lst:
                instance[x, y] = i
        self.save_im(filename, instance, layer_num)

    def center_crop(self, img, crop_x, crop_y):
        y, x = img.shape
        x0 = (x - crop_x) // 2
        y0 = (y - crop_y) // 2
        crop_img = img[y0:y0 + crop_y, x0:x0 + crop_x]
        return crop_img

    def save_im(self, name, instance, layer_num):
        print(f'Saving {name} ...')
        inst = np.array_split(instance, layer_num)  # Separates concatenated layers
        for ind, arr in enumerate(inst):
            print(arr.shape)
            output = Image.fromarray(np.uint8(arr[:-1, :]))
            filename = join(self.save_path, self.files[name][ind])
            output.save(filename)


root = ''
save = ''

process = TiffImageProcessor(root, save)
names, layered_images, num = process.handle()

for i, img in enumerate(layered_images):
    output = label(img)
    process.save_im(names[i], output, num[i])
