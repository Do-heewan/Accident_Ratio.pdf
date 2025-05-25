checkpoint_config = dict(interval=4)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = '/mnt/ext3/server1/For4_view3/best_top1_acc_epoch_18.pth'
workflow = [('train', 1), ('val', 1)]
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=4,
        speed_ratio=4,
        channel_ratio=8,
        slow_pathway=dict(
            type='resnet3d',
            depth=101,
            in_channels=4,
            pretrained=None,
            lateral=True,
            fusion_kernel=7,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            type='resnet3d',
            depth=101,
            in_channels=4,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,
        num_classes=152,
        spatial_type='avg',
        dropout_ratio=0.5),
    train_cfg=None,
    test_cfg=dict(average_clips=None))
dataset_type = 'RawframeDataset'
data_root = '/workspace/DATASET/Tmax/CarAccident/view1'
data_root_val = '/workspace/DATASET/Tmax/CarAccident/view1'
ann_file_train = '/mnt/ext3/View3/train_4th_view3.txt'
ann_file_val = '/mnt/ext3/View3/val_4th_view3.txt'
ann_file_test = '/mnt/ext3/View3/val_4th_view3.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
clip_len_setting = 32
frame_interval_setting = 1
num_clips_settting = 1
batch_size = 6
size = (480, 270)
train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=1, num_clips=1),
    dict(type='RawFrameDecode_B'),
    dict(type='Resize', scale=(480, 270)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(480, 270), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'masks', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'masks', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode_B'),
    dict(type='Resize', scale=(480, 270)),
    dict(type='Flip', flip_ratio=0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'masks', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'masks', 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode_B'),
    dict(type='Resize', scale=(480, 270)),
    dict(type='Flip', flip_ratio=0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'masks', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'masks', 'label'])
]
data = dict(
    videos_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type='RawframeDataset',
        ann_file='/mnt/ext3/View3/train_4th_view3.txt',
        data_prefix='/workspace/DATASET/Tmax/CarAccident/view1',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=1,
                num_clips=1),
            dict(type='RawFrameDecode_B'),
            dict(type='Resize', scale=(480, 270)),
            dict(type='RandomResizedCrop'),
            dict(type='Resize', scale=(480, 270), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(
                type='Collect', keys=['imgs', 'masks', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'masks', 'label'])
        ]),
    val=dict(
        type='RawframeDataset',
        ann_file='/mnt/ext3/View3/val_4th_view3.txt',
        data_prefix='/workspace/DATASET/Tmax/CarAccident/view1',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=1,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode_B'),
            dict(type='Resize', scale=(480, 270)),
            dict(type='Flip', flip_ratio=0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(
                type='Collect', keys=['imgs', 'masks', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'masks', 'label'])
        ]),
    test=dict(
        type='RawframeDataset',
        ann_file='/mnt/ext3/View3/val_4th_view3.txt',
        data_prefix='/workspace/DATASET/Tmax/CarAccident/view1',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=1,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode_B'),
            dict(type='Resize', scale=(480, 270)),
            dict(type='Flip', flip_ratio=0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(
                type='Collect', keys=['imgs', 'masks', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'masks', 'label'])
        ]))
evaluation = dict(
    interval=2,
    metrics=['top_k_accuracy', 'mean_class_accuracy'],
    top_k=(1, ),
    best_ckpt_name='best.pth')
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=1)
total_epochs = 25
work_dir = '/mnt/ext3/server1/For4_view3'
find_unused_parameters = False
gpu_ids = range(0, 4)
omnisource = False
module_hooks = []
