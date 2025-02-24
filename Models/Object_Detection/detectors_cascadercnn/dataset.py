from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS

@DATASETS.register_module()
class TrafficDataSet(CocoDataset):
    METAINFO = {
        'classes': ('vehicle', 'pedestrian', 'two-wheeled-vehicle', 'bike', 
                   'traffic-sign', 'traffic-light-red', 'traffic-light-green', 
                   'traffic-light-etc', 'crosswalk'),
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                   (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                   (0, 118, 142)]
    } 