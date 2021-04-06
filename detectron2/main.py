import copy, cv2, datetime, detectron2, json, logging, os, random, time, torch
import detectron2.data.transforms as T
import numpy as np
import pandas as pd

from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_test_loader, MetadataCatalog, DatasetCatalog, DatasetMapper
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds, setup_logger
from detectron2.utils.visualizer import Visualizer
setup_logger()

def get_dicts(img_dir, anno_path):
    dataframe = pd.read_csv(anno_path, sep=",", header=None)
    dataframe.columns = ['filename', 'x1', 'y1', 'x2', 'y2', 'class']
    image_ids = dataframe['filename'].unique()
    dataset_dicts = []

    for i, img_file in enumerate(image_ids):
        group = dataframe[dataframe['filename'] == img_file]
        annos = group[['x1', 'y1', 'x2', 'y2']].values
        objs = []
        for anno in annos:
            obj = {
                "bbox": list(anno),
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 10,
            }
            objs.append(obj)

        file_name = os.path.join(img_dir, img_file) 
        img = cv2.imread(file_name)
        height, width  = img.shape[:2]
        record = {
            'file_name': file_name,
            'image_id': i,
            'width': width,
            'height': height,
            'annotations': objs
        }
        dataset_dicts.append(record)
    return dataset_dicts


config_base = 'faster_rcnn_R_101_FPN_3x'

"""
Group 1 (Standard Training Sets):
    carla_nodr
    cut-and-paste_st
    carla_st 
    mvd_st
    coco_st 
    
Group 2 (Medium Training Sets):
    cut-and-paste20000_m carla20000_m
    mvd20_m  coco20_m  carla+mvd20_m  carla+coco20_m
    mvd120_m coco120_m carla+mvd120_m carla+coco120_m
    mvd600_m coco600_m carla+mvd600_m carla+coco600_m
"""
town = 'carla'
train_size = 'nodr' 

max_iter = 25000

output_path = 'model/%s_%s/' % (town, train_size)
os.makedirs(output_path, exist_ok=True)

test_only = False
save_output_imgs = False
train_dataset = 'carla'
test_dataset = 'coco'

home_data_dir = 'data/'
train_imgs = home_data_dir + '%s' % train_dataset
train_anno = home_data_dir + '%s/train_nodr_standard.txt' % train_dataset
test_imgs = home_data_dir + '%s' % test_dataset
test_anno = home_data_dir + '%s/test.txt' % test_dataset

target_class = 'fire_hydrant'
thing_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

DatasetCatalog.register("%s_train" % target_class, lambda :get_dicts(train_imgs, train_anno))
DatasetCatalog.register("%s_test" % target_class, lambda :get_dicts(test_imgs, test_anno))
fire_hydrant_metadata = MetadataCatalog.get("%s_test" % target_class)
MetadataCatalog.get("%s_test" % target_class).set(thing_classes=thing_classes)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/%s.yaml" % config_base))
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/%s.yaml" % config_base)
cfg.DATASETS.TRAIN = ("%s_train" % target_class,)
cfg.DATASETS.TEST = ("%s_test" % target_class,)
cfg.OUTPUT_DIR=output_path
cfg.SOLVER.MAX_ITER = max_iter
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 8

# Disable resizing
cfg.INPUT.MIN_SIZE_TRAIN = 0
cfg.INPUT.MAX_SIZE_TRAIN = 800
cfg.INPUT.MIN_SIZE_TEST = 0
cfg.INPUT.MAX_SIZE_TEST = 800

# Random cropping
cfg.INPUT.CROP = CfgNode({"ENABLED": True})
cfg.INPUT.CROP.TYPE = "relative_range"
cfg.INPUT.CROP.SIZE = [0.9, 0.9]

if not test_only:
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

try:
    os.remove(output_path+'coco_instances_results.json')
except:
    print('Did not remove coco_instances_results.json')
try:
    os.remove(output_path+'%s_test_coco_format.json' % (target_class))
except:
    print('Did not remove %s_test_coco_format.json' % (target_class))
try:
    os.remove(output_path+'%s_test_coco_format.json.lock' % (target_class))
except:
    print('Did not remove %s_test_coco_format.json.lock' % (target_class))


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    
predictor = DefaultPredictor(cfg)

if save_output_imgs:
    os.makedirs('output_imgs', exist_ok=True)
    for d in get_dicts(test_imgs, test_anno):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                    metadata=MetadataCatalog.get("%s_test" % target_class), 
                    scale=0.5
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite('output_imgs/'+d["file_name"].split('/')[-1], v.get_image()[:, :, ::-1])

evaluator = COCOEvaluator("%s_test" % target_class, ['bbox'], False, output_dir=output_path)
val_loader = build_detection_test_loader(
    get_dicts(test_imgs, test_anno), 
    mapper = DatasetMapper(cfg, is_train=False)
)
inference_on_dataset(predictor.model, val_loader, evaluator)