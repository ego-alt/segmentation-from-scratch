import re
import os.path
import numpy
import torch
from torch.utils.data import Dataset

root_dir = "../../Downloads/Summer project/BSDS300"

train_data = os.path.join(root_dir, 'images/train')
test_data = os.path.join(root_dir, 'images/test')
labels = os.path.join(root_dir, 'human/color')
humans = [folder for folder in os.listdir(labels) if not folder.startswith('.')]


def extract_labels(label_files):
    meta = {}
    data = []
    with open(label_files, 'r') as f:
        matcher = re.compile('(?P<seg>^[0-9 ]+)')
        for line in f:
            seg_match = matcher.search(line)
            if seg_match:
                string_segment = seg_match.group('seg').split(' ')
                int_segment = numpy.asarray(string_segment, dtype=int)
                data.append(int_segment)
                continue
            elif "data" not in line:
                meta_data = line.strip('\n').split(' ', 1)
                index, value = meta_data[0], meta_data[1]
                meta[index] = value
    height, width = int(meta['height']), int(meta['width'])
    # print(f"User id: {meta['user']}     Image id: {meta['image']}")
    # print(f"Height: {height}       Width: {width}")
    segmentations = numpy.zeros((height, width))
    for seg in data:
        segmentations[seg[1], seg[2]:(seg[3]+1)] = seg[0]
    return segmentations


for folder in humans:
    human_path = os.path.join(labels, folder)
    seg_files = [seg for seg in os.listdir(human_path)]
    targets = []
    for seg in seg_files:
        seg_path = os.path.join(human_path, seg)
        targets.append(extract_labels(seg_path))

"""test = os.path.join(labels, '1108/15004.seg')
extract_labels(test)"""

