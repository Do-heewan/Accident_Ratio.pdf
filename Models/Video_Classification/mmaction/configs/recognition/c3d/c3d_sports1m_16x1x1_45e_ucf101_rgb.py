_base_ = '../../_base_/models/c3d_sports1m_pretrained.py'

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/media/mingfan/DataHDD/211214_CarAccident_v1'
data_root_val = '/media/mingfan/DataHDD/211214_CarAccident_v1'

ann_file_train = '/media/mingfan/DataHDD/211214_CarAccident_v1/train-regen-new.txt'
ann_file_val = '/media/mingfan/DataHDD/211214_CarAccident_v1/val-regen-new.txt'
ann_file_test = '/media/mingfan/DataHDD/211214_CarAccident_v1/val-regen-new.txt'

split = 1  # official train/test splits. valid numbers: 1, 2, 3
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

clip_len_setting = 32
frame_interval_setting = 2
num_clips_settting = 1
batch_size = 4

train_pipeline = [
    dict(type='SampleFrames', clip_len=clip_len_setting, frame_interval=frame_interval_setting, num_clips=num_clips_settting),
    dict(type='RawFrameDecode'),
    #dict(type='Resize', scale=(128, 171)),
    dict(type='RandomCrop', size=112),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    #dict(type='Resize', scale=(128, 171)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    #dict(type='Resize', scale=(128, 171)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
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
# optimizer
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9,
    weight_decay=0.0005)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 45

checkpoint_config = dict(interval=45)
#evaluation = dict(
#    interval=10, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=2,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/c3d_sports1m_16x1x1_45e_ucf101_split_{split}_rgb/'
load_from = None
resume_from = None
workflow = [('train', 1),(('val', 1))]
