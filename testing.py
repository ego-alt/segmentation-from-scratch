import re
import os.path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

root_dir = "../../Downloads/Summer project/BSDS300"

train_data = os.path.join(root_dir, 'images/train')
test_data = os.path.join(root_dir, 'images/test')
labels = os.path.join(root_dir, 'human/color')

images = [img for img in os.listdir(train_data) or os.listdir(test_data)]  # List of training and test images
humans = [folder for folder in os.listdir(labels) if not folder.startswith('.')]
# List of separate folders containing segmentations


def extract_labels(label_files):
    """Extracts the segmentation data from the .seg files"""
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
    """Creates an image using extraced segmentation data"""
    seg_val = (seg_val / seg_max) * 255
    img = Image.fromarray(seg_val)
    img.show()


class Berkeley(Dataset):
    """Custom dataset containing train and test data & their respective labels"""
    def __init__(self, image_names, label_files):
        self.image_names = image_names
        self.labels = {}
        for user_folder in label_files:
            user_path = os.path.join(labels, user_folder)
            for seg_name in os.listdir(user_path):
                seg_path = os.path.join(user_path, seg_name)
                seg, _ = extract_labels(seg_path)
                self.labels[seg_name] = seg

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.image_names[index]
        label = self.labels[image]
        sample = {"Image": image, "Label": label}
        return sample


DS = Berkeley(images, humans)

"""test = os.path.join(labels, '1115/66053.seg')
segmentation, seg_num = extract_labels(test)
create_image(segmentation, seg_num)"""
