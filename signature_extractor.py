# COMMON LIBRARIES
import os
import cv2

from datetime import datetime

# DATA SET PREPARATION AND LOADING
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

# VISUALIZATION
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

# CONFIGURATION
from detectron2 import model_zoo
from detectron2.config import get_cfg

# EVALUATION
from detectron2.engine import DefaultPredictor

# TRAINING
from detectron2.engine import DefaultTrainer

import numpy as np
import cv2

# HYPERPARAMETERS
ARCHITECTURE = "mask_rcnn_R_101_FPN_3x"
CONFIG_FILE_PATH = f"COCO-InstanceSegmentation/{ARCHITECTURE}.yaml"
MAX_ITER = 3000
EVAL_PERIOD = 200
BASE_LR = 0.001
NUM_CLASSES = 3

train_images_path = "C:/Users/Arpit/Downloads/Signature Extractor.v1i.coco/train"
test_images_path = "C:/Users/Arpit/Downloads/Signature Extractor.v1i.coco/test"
val_images_path = "C:/Users/Arpit/Downloads/Signature Extractor.v1i.coco/valid"

train_annotation_path = "C:/Users/Arpit/Downloads/Signature Extractor.v1i.coco/annotations/train_annotations.coco.json"
test_annotation_path = "C:/Users/Arpit/Downloads/Signature Extractor.v1i.coco/annotations/test_annotations.coco.json"
val_annotation_path = "C:/Users/Arpit/Downloads/Signature Extractor.v1i.coco/annotations/valid_annotations.coco.json"

train_dataset_name = "train_dataset"
test_dataset_name = "test_dataset"
val_dataset_name = "valid_dataset"

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
from detectron2.data.datasets import register_coco_instances
register_coco_instances(train_dataset_name, {}, train_annotation_path, train_images_path)
register_coco_instances(test_dataset_name, {}, test_annotation_path, test_images_path)
register_coco_instances(val_dataset_name, {}, val_annotation_path, val_images_path)

metadata = MetadataCatalog.get(train_dataset_name)
dataset_train = DatasetCatalog.get(train_dataset_name)

class Detector:
    def __init__(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE_PATH))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_FILE_PATH)
        self.cfg.DATASETS.TRAIN = (train_dataset_name,)
        self.cfg.DATASETS.TEST = (test_dataset_name,)
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.INPUT.MASK_FORMAT='bitmask'
        self.cfg.SOLVER.BASE_LR = BASE_LR
        self.cfg.SOLVER.MAX_ITER = MAX_ITER
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
        self.cfg.MODEL.DEVICE = 'cpu'
        #Load model config and pretrained model
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.DEVICE = 'cpu'
        self.cfg.MODEL.WEIGHTS = os.path.join("model_final.pth")
        self.predictor = DefaultPredictor(self.cfg)

    def onImage_path(self, image_path):
        image = cv2.imread(image_path)
        predictions = self.predictor(image)
        viz = Visualizer(image[:,:,::-1], metadata = metadata, instance_mode = ColorMode.IMAGE_BW)
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        return output
        
    def onImage(self, image):
        if(image is None): 
            return None
        predictions = self.predictor(image)
        # viz = Visualizer(image[:,:,::-1], metadata = metadata, instance_mode = ColorMode.IMAGE_BW)
        # output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        return predictions