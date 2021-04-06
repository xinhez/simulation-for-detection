#@title
from detectron2.utils.logger import create_small_table
# from detectron2.evaluation.coco_evaluation import *
import contextlib
import copy
import cv2
import io
import itertools
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
import detectron2.utils.comm as comm

from detectron2.engine import DefaultTrainer
from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
from detectron2.data import build_detection_test_loader

from detectron2 import model_zoo
import detectron2.utils.comm as comm
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper   # the default mapper
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, CfgNode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures.boxes import BoxMode


config_base = 'faster_rcnn_R_101_FPN_3x'

home_data_dir = 'data/'
test_dataset = 'pitt'
test_imgs = home_data_dir + test_dataset
test_anno = home_data_dir + '%s/test.txt' % test_dataset
save_output_imgs = True
score_thresh_test = 0

os.makedirs('output', exist_ok=True)
fw = open('output/log_%s_st' % test_dataset, 'w')
output_path = 'output_%s_st' % (test_dataset)

experiments = [
    # 'cut-and-paste_st',
    # 'carla_nodr',
    'carla_st',
    # 'mvd_st', # 03100
    # 'coco_st', # 25000
    ]

labels = [
        # 'cut-and-paste',
        # 'carla without dr',
        'carla',
        # 'mvd',
        # 'coco',

    ]


target_class = 'fire_hydrant'
thing_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/%s.yaml" % config_base))
cfg.DATASETS.TEST = ("%s_test" % target_class,)
cfg.SOLVER.MAX_ITER = 25000
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 8

# Disable resizing
cfg.INPUT.MIN_SIZE_TRAIN = 0
cfg.INPUT.MAX_SIZE_TRAIN = 800
cfg.INPUT.MIN_SIZE_TEST = 0
cfg.INPUT.MAX_SIZE_TEST = 2000

# Random cropping
cfg.INPUT.CROP = CfgNode({"ENABLED": True})
cfg.INPUT.CROP.TYPE = "relative_range"
cfg.INPUT.CROP.SIZE = [0.9, 0.9]

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

DatasetCatalog.register("%s_test" % target_class, lambda :get_dicts(test_imgs, test_anno))
fire_hydrant_metadata = MetadataCatalog.get("%s_test" % target_class)
MetadataCatalog.get("%s_test" % target_class).set(thing_classes=thing_classes)


class COCOEvaluatorCustom(COCOEvaluator):
    # inspired from Detectron:
    # https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L255 # noqa
    def _evaluate_box_proposals(dataset_predictions, coco_api, thresholds=None, area="all", limit=None):
        """
        Evaluate detection proposal recall metrics. This function is a much
        faster alternative to the official COCO API recall evaluation code. However,
        it produces slightly different results.
        """
        # Record max overlap value for each gt box
        # Return vector of overlap values
        print(area)
        areas = {
            "all": 0,
            "small": 1,
            "medium": 2,
            "large": 3,
            "96-128": 4,
            "128-256": 5,
            "256-512": 6,
            "512-inf": 7,
        }
        area_ranges = [
            [0 ** 2, 1e5 ** 2],  # all
            [0 ** 2, 32 ** 2],  # small
            [32 ** 2, 96 ** 2],  # medium
            [96 ** 2, 1e5 ** 2],  # large
            [96 ** 2, 128 ** 2],  # 96-128
            [128 ** 2, 256 ** 2],  # 128-256
            [256 ** 2, 512 ** 2],  # 256-512
            [512 ** 2, 1e5 ** 2],
        ]  # 512-inf
        assert area in areas, "Unknown area range: {}".format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = []
        num_pos = 0

        for i, prediction_dict in enumerate(dataset_predictions):
            predictions = prediction_dict["proposals"]

            # sort predictions in descending order
            # TODO maybe remove this and make it explicit in the documentation
            inds = predictions.objectness_logits.sort(descending=True)[1]
            predictions = predictions[inds]

            ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
            anno = coco_api.loadAnns(ann_ids)
            gt_boxes = [
                BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                for obj in anno
                if obj["iscrowd"] == 0
            ]
            gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
            gt_boxes = Boxes(gt_boxes)
            gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

            if len(gt_boxes) == 0 or len(predictions) == 0:
                continue
            valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
            gt_boxes = gt_boxes[valid_gt_inds]
            if gt_boxes:
                print(i)
            num_pos += len(gt_boxes)

            if len(gt_boxes) == 0:
                continue

            if limit is not None and len(predictions) > limit:
                predictions = predictions[:limit]

            overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)

            _gt_overlaps = torch.zeros(len(gt_boxes))
            for j in range(min(len(predictions), len(gt_boxes))):
                # find which proposal box maximally covers each gt box
                # and get the iou amount of coverage for each gt box
                max_overlaps, argmax_overlaps = overlaps.max(dim=0)

                # find which gt box is 'best' covered (i.e. 'best' = most iou)
                gt_ovr, gt_ind = max_overlaps.max(dim=0)
                assert gt_ovr >= 0
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the iou coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert _gt_overlaps[j] == gt_ovr
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1

            # append recorded iou coverage level
            gt_overlaps.append(_gt_overlaps)
        gt_overlaps = (
            torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(0, dtype=torch.float32)
        )
        gt_overlaps, _ = torch.sort(gt_overlaps)

        if thresholds is None:
            step = 0.05
            thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
        recalls = torch.zeros_like(thresholds)
        # compute recall for each iou threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
        # ar = 2 * np.trapz(recalls, thresholds)
        ar = recalls.mean()
        return {
            "ar": ar,
            "recalls": recalls,
            "thresholds": thresholds,
            "gt_overlaps": gt_overlaps,
            "num_pos": num_pos,
        }
      
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        self.coco_eval = coco_eval
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]
        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}
        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Note that some metrics cannot be computed.")
        if class_names is None or len(class_names) < 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]
        results_per_category = []
        idx = 0 
        precision = precisions[:, :, thing_classes.index('fire hydrant'), 0, -1]
        
        x = np.linspace(0, 1, 101)
        #precision[0, :] means at IOU 0.5 and at all of the recall
        #precision[-1, :] means IOU at .95
        self.pr = precision[idx, :]
        plt.plot(x, precision[idx, :])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('test' + ' -- IOU = ' + str(idx*0.05 + 0.5))
        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)
        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results

def inference(model_path):
    cfg.MODEL.WEIGHTS = os.path.join("./", model_path)  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh_test   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluatorCustom("%s_test" % target_class, cfg, False, output_dir="./%s/" % output_path)
    mapper = DatasetMapper(cfg, is_train=False)
    val_loader = build_detection_test_loader(DatasetCatalog.get("%s_test" % target_class), mapper = mapper)
    result = inference_on_dataset(predictor.model, val_loader, evaluator)['bbox']
    line = '%4.1f & %4.1f & %4.1f & %4.1f & %4.1f & %4.1f' % (result['AP'], result['AP50'], result['AP75'], result['APs'], result['APm'], result['APl'])
    print(line)
    fw.write(line + '\n')


    if save_output_imgs:
        os.makedirs('%s_output_imgs' % (model_path.split('/')[1]), exist_ok=True)
        for d in get_dicts(test_imgs, test_anno):
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                        metadata=MetadataCatalog.get("%s_test" % target_class), 
                        scale=0.5
            )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite('%s_output_imgs/' % (model_path.split('/')[1])+d["file_name"].split('/')[-1], v.get_image()[:, :, ::-1])
    return evaluator.pr

try:
    os.remove('%s/instances_predictions.pth' % output_path)
except:
    print('Did not remove instances_predictions.pth')
try:
    os.remove('%s/coco_instances_results.json' % output_path)
except:
    print('Did not remove coco_instances_results.json')
try:
    os.remove('%s/%s_test_coco_format.json' % (output_path, target_class))
except:
    print('Did not remove %s_test_coco_format.json' % (target_class))
try:
    os.remove('%s/%s_test_coco_format.json.lock' % (output_path, target_class))
except:
    print('Did not remove %s_test_coco_format.json.lock' % (target_class))

out = np.zeros((len(experiments), 101))
for i, exp in enumerate(experiments):
    fw.write(exp + ' ')
    out[i, :] = inference('model/'+exp + '/model_final.pth')
    print('=' * 20, 'processed', exp)

x = np.linspace(0, 1, 101)
for i, label in enumerate(labels):
    plt.plot(x, out[i, :], label=label)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.title('test' + ' -- IOU = ' + str(0.5))
plt.savefig('output/pr_%s_st.png' % test_dataset)
plt.show()

fw.close()

