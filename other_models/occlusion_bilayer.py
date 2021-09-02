import os
import cv2

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

root_dir = '/Users/batfolder/Downloads/Summer project'


def get_images(img_path, lbl_path):
    img_folder = os.path.join(root_dir, img_path)
    lbl_folder = os.path.join(root_dir, lbl_path)

    images = sorted(os.listdir(img_folder))[1:]
    labels = sorted(os.listdir(lbl_folder))

    for idx, filename in enumerate(images):
        img_file = os.path.join(img_folder, filename)
        height, width = cv2.imread(img_file).shape[:2]

        record = {"file_name": img_file,
                  "image_id": idx,
                  "height": height,
                  "width": width}

        lbl_file = os.path.join(lbl_folder, labels[idx])


get_images('MP6843_img_full', 'MP6843_inst')
