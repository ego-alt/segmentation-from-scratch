import re
import os.path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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


class CropArrayCentre(object):
    """Custom transform to crop the centre of arrays (both images and segmentations)"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        y, x, _ = image.shape
        crop_y, crop_x = self.output_size
        start_x, start_y = x // 2 - (crop_x // 2), y // 2 - (crop_y // 2)
        image = image[start_y: start_y + crop_y, start_x: start_x + crop_x]
        label = label[start_y: start_y + crop_y, start_x: start_x + crop_x]
        return {'image': image, 'label': label}


class TwoTensor(object):
    """Custom transform to convert arrays (both images and segmentations) to tensors"""

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label)}


class Berkeley(Dataset):
    """Custom dataset containing training/test data + their respective labels"""

    def __init__(self, image_files, label_files):
        """Images and labels are converted into np.arrays and listed in ascending index
        :param image_files: Path to images
        :param label_files: Path to segmentation labels"""
        self.images, self.labels = self.array_from_path(image_files, label_files)
        self.transform = transforms.Compose([
            CropArrayCentre(321),  # Crops image + segmentation to uniform size
            TwoTensor()  # Converts image + segmentation np.array to torch.tensor
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = {"image": self.images[index], "label": self.labels[index]}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def array_from_path(self, image_files, label_files):
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

        images = [np.asarray(Image.open(os.path.join(image_files, img))) for img in images]
        labels = [value for _, value in sorted(ordered_files.items(), key=lambda ele: ele[0])]
        return images, labels


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Encoder(torch.nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = in_channel * 2
        self.stack = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channel, self.out_channel, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.out_channel, self.out_channel, (3, 3)),
        )

    def forward(self, x):
        self.stack(x)


class Decoder(torch.nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = in_channel / 2
        self.stack = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channel, self.out_channel, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.out_channel, self.out_channel, (3, 3))
        )

    def forward(self):
        pass
