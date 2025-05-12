# Copyright (c) OpenMMLab. All rights reserved.
import sys
import os
from mmdet.registry import DATASETS
from .coco import CocoDataset

# 현재 파일의 상위 디렉토리를 찾아 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

@DATASETS.register_module()
class TrafficDataSet(CocoDataset):
    """Dataset for DeepFashion."""

    METAINFO = {
        'classes': ("two-wheeled-vehicle","traffic-light-red","pedestrian",
                    "crosswalk","bike","traffic-light-green","vehicle",
                    "traffic-light-etc", "traffic-sign",),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(0, 192, 64), (0, 64, 96), (128, 192, 192), (0, 64, 64),
                    (0, 192, 224), (0, 192, 192), (128, 192, 64), (0, 192, 96),
                    (128, 32, 192),]
    }
