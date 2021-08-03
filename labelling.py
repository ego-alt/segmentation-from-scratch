import numpy as np


class Cluster:
    def __init__(self, array):
        cells = np.where(array > 0.5)
        full_x, full_y = cells

        self.cells = set(zip(full_x, full_y))
        self.instance = np.zeros_like(array)
        self.cell_clusters = []

    def find_neigh(self, col, row, cluster):
        x1 = (col - 1) if col > 0 else col
        x2 = (col + 1) if col <= len(self.cells) else col
        y1 = (row - 1) if row > 0 else row
        y2 = (row + 1) if row <= len(self.cells) else row

        """Neighbours: surrounding 8 pixels & myself
        Friends: neighbouring cells"""
        neigh = [(c, r) for c in range(x1, x2 + 1) for r in range(y1, y2 + 1)]
        friends = (self.cells & set(neigh))
        if friends: cluster.extend(list(friends))

        self.cells = self.cells.difference(friends)  # Remove neighbours from consideration
        for (c, r) in friends:
            self.find_neigh(c, r, cluster)
        return cluster

    def main(self):
        for (x, y) in self.cells:
            cluster = []
            cluster = self.find_neigh(x, y, cluster)  # Find surrounding neighbours
            if cluster: self.cell_clusters.append(cluster)

    def save_im(self):
        for idx, lst in enumerate(self.cell_clusters):
            for (x, y) in lst:
                self.instance[x - 1, y - 1] = idx + idx
