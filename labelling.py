import numpy as np
from PIL import Image


class Cluster:
    def __init__(self, array):
        cells = np.where(array > 0.5)
        full_x, full_y = cells

        self.cells = set(zip(full_x, full_y))
        self.c_num = len(self.cells)
        self.instance = np.zeros_like(array)
        self.cell_clusters = []

    def main(self):
        for (x, y) in self.cells:
            cluster = []
            cluster = self.find_cell(x, y, cluster)  # Find surrounding neighbours
            if cluster: self.cell_clusters.append(cluster)

        self.clustering()
        self.save_im()

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

    def save_im(self):
        for idx, lst in enumerate(self.cell_clusters):
            for (x, y) in lst:
                self.instance[x - 1, y - 1] = idx

        output = Image.fromarray(np.uint8(self.instance))
        output.save()
