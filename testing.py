import re
import os.path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

root_dir = "../../Downloads/Summer project/BSDS300"

train_data = os.path.join(root_dir, 'images/train')
test_data = os.path.join(root_dir, 'images/test')
labels = os.path.join(root_dir, 'human/color')


def extract_labels(label_files):
    """Converts segmentation data from .seg files into np.array"""
    meta = {}
    data = []
    with open(label_files, 'r') as f:
        matcher = re.compile('(?P<seg>^[0-9 ]+)')
        for line in f:
            seg_match = matcher.search(line)
            if seg_match:
                string_segment = seg_match.group('seg').split(' ')
                int_segment = np.asarray(string_segment, dtype=int)
                data.append(int_segment)
                continue
            elif "data" not in line:
                meta_data = line.strip('\n').split(' ', 1)
                index, value = meta_data[0], meta_data[1]
                meta[index] = value
    height, width = int(meta['height']), int(meta['width'])
    seg_num = int(meta['segments'])
    # print(f"User id: {meta['user']}     Image id: {meta['image']}")
    # print(f"Height: {height}       Width: {width}")
    segmentation = np.zeros((height, width))
    for seg in data:
        segmentation[seg[1], seg[2]:(seg[3] + 1)] = seg[0]
    return segmentation, seg_num


def create_image(seg_val, seg_max):
    """Creates an image using extracted segmentation data"""
    seg_val = (seg_val / seg_max) * 255
    img = Image.fromarray(seg_val)
    img.show()


class Berkeley(Dataset):
    """Custom dataset containing training or test data & their respective labels"""

    def __init__(self, image_files, label_files, transform=None):
        """Images and labels are converted into np.arrays and listed in ascending index
        :param image_files: Path to images
        :param label_files: Path to segmentation labels"""
        images = sorted([img for img in os.listdir(image_files)])  # List in format .jpg
        image_names = [os.path.splitext(img)[0] for img in images]  # List of image ids (sans .jpg)
        ordered_files = {}
        for root, user_folder, files in os.walk(label_files):
            for file in files:
                file_name = os.path.splitext(file)[0]
                if file_name in image_names:
                    file_path = os.path.join(root, file)
                    seg, _ = extract_labels(file_path)
                    ordered_files[file_name] = seg

        self.images = [np.asarray(Image.open(os.path.join(image_files, img))) for img in images]
        self.labels = [value for _, value in sorted(ordered_files.items(), key=lambda ele: ele[0])]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        sample = {"Image": image, "Label": label}
        return sample


testset = Berkeley(test_data, labels)
testloader = DataLoader(testset, batch_size=25, shuffle=False)

"""test = os.path.join(labels, '1115/66053.seg')
segmentation, seg_num = extract_labels(test)
create_image(segmentation, seg_num)"""
