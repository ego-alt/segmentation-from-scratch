import os
import cv2
import random
import numpy as np
from instance_seg.train import Match, find_masks, find_boxes

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

labels = '/Users/batfolder/Downloads/Summer project/MP6843_inst'
images = '/Users/batfolder/Downloads/Summer project/MP6843_img_full'


def get_image_dicts(lbl_dir, img_dir, d):
    """Fits data into detectron2's standard format"""
    handler = Match(lbl_dir, img_dir)
    w1, labels3d = handler.main('w1')
    if d == "train":
        w1, labels3d = w1[0:70], labels3d[0:70]
    if d == "valid":
        w1, labels3d = w1[70:90], labels3d[70:90]

    dataset_dicts = []
    for idx, img in enumerate(w1):
        height, width = img.shape[:2]
        filename = os.path.join(img_dir, handler.names[idx] + "w1.TIF")
        record = {
            "file_name": filename,
            "image_id": idx,
            "height": height,
            "width": width
        }
        _, _, stack_num = labels3d[idx].shape  # Number of layers
        masks, mask_num = find_masks(stack_num, labels3d[idx])
        boxes = find_boxes(mask_num, masks)

        objs = []
        for i in range(len(masks)):
            mask = [(x + 0.5, y + 0.5) for y, x in zip(*np.nonzero(masks[i]))]
            mask = [p for x in mask for p in x]
            obj = {
                "bbox": boxes[i],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [mask],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


class BCNet:
    def __init__(self, lbl_dir, img_dir):
        self.lbl_dir = lbl_dir
        self.img_dir = img_dir
        self.register()
        self.cfg = get_cfg()  # Loads default configurations

    def register(self):
        for d in ["train", "val"]:
            DatasetCatalog.register("cells_" + d, lambda d=d: get_image_dicts(self.lbl_dir, self.img_dir, d))
            MetadataCatalog.get("cells_" + d).set(thing_classes=["cell"])

    def train(self):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    def view_image(self):
        """Designed for visualisation on the local computer, not Colab"""
        cell_metadata = MetadataCatalog.get("cells_train")
        dataset_dicts = get_image_dicts(self.lbl_dir, self.img_dir, "train")
        for d in random.sample(dataset_dicts, 3):
            img = cv2.imread(d["file_name"])
            img = cv2.resize(img, (696, 520), interpolation=cv2.INTER_AREA)
            visualizer = Visualizer(img[:, :, ::-1], metadata=cell_metadata, scale=0.5)
            out = visualizer.draw_dataset_dict(d)

            cv2.namedWindow('window', cv2.WINDOW_KEEPRATIO)
            cv2.imshow("window", out.get_image()[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

