data = dict(
    train=dict(
        type='RawframeDataset',
        ann_file='data/kinetics400/kinetics_train_list_rawframes.txt',
        data_prefix='data/kinetics400/rawframes_train',
        pipeline=[
            dict(type='RawFrameDecode_B'),  # 커스텀 파이프라인 적용
            dict(type='Resize', scale=(-1, 256)),
            dict(type='RandomResizedCrop'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='ToTensor'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        ]
    )
)
