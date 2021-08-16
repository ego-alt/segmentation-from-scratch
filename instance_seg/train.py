import cv2
import re
import numpy as np
from os.path import join
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.optim.lr_scheduler import OneCycleLR
from engine import train_one_epoch, evaluate

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class ArrayMaker:
    """Handles image datasets by resizing, cropping and converting to arrays"""

    def __init__(self, root_path):
        self.root = root_path
        self.files = {}  # Directory for image versions/ layers
        self.arrdict = {}  # Holds image arrays

        self.org_files()

    def main(self, dim=None, crop=None, greyscale=False):
        for name in self.files:
            for i in self.files[name]:
                if greyscale:
                    im = cv2.imread(join(self.root, i), 0)
                else:
                    im = cv2.imread(join(self.root, i))
                if dim: im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

                im = np.array(im)
                if crop:  # Center crop
                    y, x, *_ = im.shape
                    x0 = (x - crop) // 2
                    y0 = (y - crop) // 2
                    im = im[x0:x0 + crop, y0:y0 + crop]
                self.listdict(self.arrdict, name, im)

    def org_files(self):
        """Image variants can be accessed under the same key in a dictionary"""
        regex = "^F0[1-4]_[0-9]+"  # Filename shared between versions/ layers
        for file in sorted(listdir(self.root)):
            if not file.startswith('.'):
                filename = re.findall(regex, file)[0]
                self.listdict(self.files, filename, file)

    def common_elements(self, other):
        """Ensures images and labels can be paired up properly"""
        stored = {k: self.arrdict[k] for k in self.arrdict if k in other}
        self.arrdict = stored

    def filtering(self, keyword):
        """Filters images given specific criteria in their filenames"""
        filtered = []
        for name in self.files:
            filename = self.files[name]
            f = [filename.index(i) for i in filename if keyword in i]
            filtered.extend(self.arrdict[name][i] for i in f)
        return filtered  # List of filtered image arrays

    def stacking(self):
        """Stacks different layers of an image into a single array"""
        stacked = [np.stack(self.arrdict[name], axis=-1) for name in self.arrdict]
        return stacked  # List of 3D segmentation arrays

    def listdict(self, dictionary, key, value):
        """Updates a dictionary of lists"""
        if key not in dictionary:
            dictionary[key] = list()
        dictionary[key].append(value)


class CellImages(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ind):
        img = self.transform(self.images[ind])
        lbl = self.labels[ind]
        _, _, stack_num = lbl.shape  # Number of layers

        masks = None
        for i in range(stack_num):
            # Iterates over layers in a 3D segmentation
            layer = lbl[:, :, i]
            obj_ids = np.unique(layer)[1:]  # Excludes the background
            # Set of binary masks for each cell
            if masks is None:
                masks = layer == obj_ids[:, None, None]
            else:
                current = layer == obj_ids[:, None, None]
                masks = np.concatenate((masks, current), axis=0)
        mask_num, _, _ = masks.shape  # Number of unique cells

        boxes = []
        # Locates bounding boxes for each cell
        for i in range(mask_num):
            coord = np.where(masks[i])  # Coordinates for "is cell"
            x0, x1 = np.min(coord[1]), np.max(coord[1])
            y0, y1 = np.min(coord[0]), np.max(coord[0])
            if x0 < x1 and y0 < y1:
                boxes.append([x0, y0, x1, y1])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)  # Box coordinates
        labels = torch.ones((mask_num,), dtype=torch.int64)  # Only 1 class
        masks = torch.as_tensor(masks, dtype=torch.uint8)  # Segmentation masks
        image_id = torch.as_tensor([ind])  # Unique image identifier
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((mask_num,), dtype=torch.int64)

        # Required for the pretrained Mask R-CNN
        target = {"boxes": boxes,
                  "labels": labels,
                  "masks": masks,
                  "image_id": image_id,
                  "area": area,
                  "iscrowd": iscrowd}

        return img, target


def instance(num_classes):
    """Load an instance segmentation model pretrained on COCO"""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # Number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pretrained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    hidden_layer = 256
    # Number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # Replace the pretrained mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


class InitModel:
    def __init__(self, num_classes=2):
        self.model = instance(num_classes)
        self.model.to(device)

    def main(self, train, test, epochs=30):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimiser = torch.optim.SGD(params, lr=0.005, weight_decay=0.001)
        scheduler = OneCycleLR(optimiser, max_lr=0.05, steps_per_epoch=len(train), epochs=epochs)
        for epoch in range(epochs):
            train_one_epoch(self.model, optimiser, train, device, epoch, print_freq=10)
            scheduler.step()
            evaluate(self.model, test, device=device)


class ImageTest:
    def __init__(self, test_img):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.img = test_img
        self.classes = ['__background__', 'cell']

    def main(self, model, conf=0.5):
        figure = plt.figure(figsize=(15,10))

        for ind, i in enumerate(self.img):
            i = self.transform(i).to(device)
            boxes, classes, masks = self.run_model(model, i, conf)
            ax = figure.add_subplot(2, 2, ind + 1)
            image = self.from_gpu(i).transpose(1,2,0)
            image = Image.fromarray(np.uint8(image * 255)).convert('L')
            ax.imshow(image, cmap='gray')
            self.show_boxes(boxes, ax)
            self.show_masks(masks, ax)

    def show_boxes(self, boxes, ax):
        for j in range(len(boxes)):
            x, y = boxes[j][0]
            x1, y1 = boxes[j][1]
            width, height = x1 - x, y1 - y
            ax.add_patch(patches.Rectangle(
                (x, y), width, height,
                linewidth=1.5, edgecolor='r', facecolor='none'
            ))

    def show_masks(self, masks, ax):
        masks = np.multiply(masks, 1)
        for j in range(len(masks)):
            masks[j] = masks[j] * (j + 1)
        masks = sum(masks)
        masks = Image.fromarray(np.uint8(masks))
        ax.imshow(masks, cmap='jet', alpha=0.5)

    def run_model(self, model, img, conf):
        model.eval()
        pred = model([img])
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        pred_conf = [pred_score.index(x) for x in pred_score if x > conf][-1]

        boxes = self.from_gpu(pred[0]['boxes'])
        boxes = [[(i[0], i[1]), (i[2], i[3])] for i in boxes]
        labels = [self.classes[i] for i in self.from_gpu(pred[0]['labels'])]
        masks = self.from_gpu((pred[0]['masks'] > 0.5).squeeze())

        pred_boxes = boxes[:pred_conf + 1]
        pred_class = labels[:pred_conf + 1]
        pred_masks = masks[:pred_conf + 1]

        return pred_boxes, pred_class, pred_masks

    def from_gpu(self, tensor):
        return tensor.detach().cpu().numpy()
