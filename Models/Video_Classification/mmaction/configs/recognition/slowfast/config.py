_base_ = [
    '../../_base_/default_runtime.py'
]

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=4,  # tau
        speed_ratio=4,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            type='resnet3d',
            depth=101,
            in_channels = 4,
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
            in_channels = 4,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,  # 2048+256
        num_classes=60,
        spatial_type='avg',
        dropout_ratio=0.5))

######################
# 데이터셋 수정 #######
######################
dataset_type = 'RawframeDataset'
data_root = '/workspace/DATASET/Tmax/CarAccident/view1'
data_root_val = '/workspace/DATASET/Tmax/CarAccident/view1'

ann_file_train = '/mnt/ext3/View1/train_2nd_view1.txt'
ann_file_val = '/mnt/ext3/View1/val_2nd_view1.txt'
ann_file_test = '/mnt/ext3/View1/val_2nd_view1.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

clip_len_setting = 32
frame_interval_setting = 1
num_clips_settting = 1
batch_size = 6
size = (480, 270)
train_pipeline = [
    dict(type='SampleFrames', clip_len=clip_len_setting, frame_interval=frame_interval_setting, num_clips=num_clips_settting),
    #dict(type='SampleFrames', clip_len=clip_len_setting, frame_uniform=True),
    dict(type='RawFrameDecode_B'),
    dict(type='Resize', scale=size),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=size, keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'masks', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'masks', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=clip_len_setting,
        frame_interval=frame_interval_setting,
        num_clips=num_clips_settting,
        test_mode=True),
    dict(type='RawFrameDecode_B'),
    dict(type='Resize', scale=size),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'masks',  'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'masks','label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=clip_len_setting,
        frame_interval=frame_interval_setting,
        num_clips=num_clips_settting,
        test_mode=True),
    dict(type='RawFrameDecode_B'),
    dict(type='Resize', scale=size),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'masks','label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'masks','label'])
]
data = dict(
    videos_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

evaluation = dict(interval=2, metrics=['top_k_accuracy', 'mean_class_accuracy'], top_k=(1,), best_ckpt_name='best.pth')

# optimizer
optimizer = dict(
    type='SGD', lr=0.002, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=1)

total_epochs = 25

# runtime settings
checkpoint_config = dict(interval=4)
#work_dir = '/mnt/ext3/server2/For2_view1'
work_dir = '/home/For2_view1'
find_unused_parameters = False
load_from = '/home/weight/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb_20200728-145f1097.pth'
workflow = [('train', 1), ('val', 1)]
# workflow = [('val', 1), ('train', 1)]