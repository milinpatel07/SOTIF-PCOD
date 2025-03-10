
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union
from sklearn.metrics import precision_score, recall_score, f1_score
import mmengine
import numpy as np
import torch
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmdet3d.evaluation import kitti_eval
from mmdet3d.registry import METRICS
from mmdet3d.structures import (Box3DMode, CameraInstance3DBoxes,
                                LiDARInstance3DBoxes, points_cam2img)
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector
from mmdet3d.apis import train_detector, single_gpu_test
from mmdet3d.datasets import build_dataloader
from mmengine import load_checkpoint
from mmengine.runner import load_checkpoint
from mmengine.apis import single_gpu_test
from mmdet3d.evaluation import KittiDataset, KittiEvaluator

# Load your model and test data
# Note: Replace these paths with the actual paths to your config file, checkpoint file, and test data
config_file = 'configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py'
checkpoint_file = 'checkpoints/second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth'
test_data = 'data/kitti'

# Build the model
model = build_detector(
    config_file,
    train_cfg=None,
    test_cfg=None)

# Load checkpoint
load_checkpoint(model, checkpoint_file)

# Build the dataset and dataloader
dataset = build_dataset(config.model.train_pipeline.dataset)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=1,
    dist=False,
    shuffle=False)

# Run the test
outputs = single_gpu_test(model, data_loader)

# Create a KittiEvaluator instance
evaluator = KittiEvaluator('data/kitti/kitti_infos_val.pkl')

# Compute metrics
metrics = evaluator.compute_metrics(outputs)

# Print the metrics
print("Precision:", metrics['precision'])
print("Recall:", metrics['recall'])
print("F1-score:", metrics['f1_score'])
print("mAP:", metrics['mAP'])
