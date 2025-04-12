from mmcv import Config
cfg = Config.fromfile('./configs/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb.py')
from mmcv.runner import set_random_seed

# Modify dataset type and path
cfg.dataset_type = 'RawframeDataset'
cfg.data_root = '/media/mingfan/DATASSD/Video-Swin-Transformer/kinetics400_tiny/train/rawframes'
cfg.data_root_val = '/media/mingfan/DATASSD/Video-Swin-Transformer/kinetics400_tiny/val/rawframes'
cfg.ann_file_train = '/media/mingfan/DATASSD/Video-Swin-Transformer/kinetics400_tiny/kinetics_tiny_train_frame.txt'
cfg.ann_file_val = '/media/mingfan/DATASSD/Video-Swin-Transformer/kinetics400_tiny/kinetics_tiny_val_frame.txt'
cfg.ann_file_test = '/media/mingfan/DATASSD/Video-Swin-Transformer/kinetics400_tiny/kinetics_tiny_val_frame.txt'

cfg.data.test.type = 'RawframeDataset'
cfg.data.test.ann_file = 'kinetics400_tiny/kinetics_tiny_val_frame.txt'
cfg.data.test.data_prefix = 'kinetics400_tiny/val/rawframes/'

cfg.data.train.type = 'RawframeDataset'
cfg.data.train.ann_file = '/media/mingfan/DATASSD/Video-Swin-Transformer/kinetics400_tiny/kinetics_tiny_train_frame.txt'
cfg.data.train.data_prefix = '/media/mingfan/DATASSD/Video-Swin-Transformer/kinetics400_tiny/train/rawframes/'

cfg.data.val.type = 'RawframeDataset'
cfg.data.val.ann_file = '/media/mingfan/DATASSD/Video-Swin-Transformer/kinetics400_tiny/kinetics_tiny_val_frame.txt'
cfg.data.val.data_prefix = '/media/mingfan/DATASSD/Video-Swin-Transformer/kinetics400_tiny/val/rawframes/'

# The flag is used to determine whether it is omnisource training
cfg.setdefault('omnisource', False)
# Modify num classes of the model in cls_head
cfg.model.cls_head.num_classes = 2
# We can use the pre-trained TSN model
cfg.load_from = '/media/mingfan/DATASSD/Video-Swin-Transformer/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './tutorial_exps'
img_norm_cfg = dict(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False)

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.data.videos_per_gpu = cfg.data.videos_per_gpu // 16
cfg.optimizer.lr = cfg.optimizer.lr / 8 / 16
cfg.total_epochs = 10

# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 5
# We can set the log print interval to reduce the the times of printing log
cfg.log_config.interval = 5

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# Save the best
cfg.evaluation.save_best='auto'


# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')



#Train a new recognizer
import os.path as osp

from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.apis import train_model

import mmcv

# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the recognizer
model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_model(model, datasets, cfg, distributed=False, validate=True)


# Tutorial 3: Adding New Dataset
# https://github.com/open-mmlab/mmaction2/blob/master/docs/tutorials/3_new_dataset.md#an-example-of-a-custom-dataset