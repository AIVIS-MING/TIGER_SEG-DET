img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile_Tiger'),
    dict(type='LoadAnnotations__Tiger', reduce_zero_label=False),
    #dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile_Tiger'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


dataset_type = 'TIGERDataset'
data_root = '/workspace/DATASET/TIGER-Dataset/wsirois/roi-level-annotations/tissue-bcss'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='TXT_File/train_1_fold_512_4x.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='TXT_File/val_1_fold_512_4x.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='TXT_File/val_1_fold_512_4x.txt',
        pipeline=test_pipeline))

# model settings
#norm_cfg = dict(type='BN', requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='BiSeNetV2',
        detail_channels=(64, 64, 128),
        semantic_channels=(16, 32, 64, 128),
        semantic_expansion_ratio=6,
        bga_channels=128,
        out_indices=(0, 1, 2, 3, 4),
        init_cfg=None,
        align_corners=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=0,
        channels=1024,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=16,
            channels=16,
            num_convs=2,
            num_classes=8,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=32,
            channels=64,
            num_convs=2,
            num_classes=8,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=64,
            channels=256,
            num_convs=2,
            num_classes=8,
            in_index=3,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=1024,
            num_convs=2,
            num_classes=8,
            in_index=4,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))



# optimizer Setting==========================
# optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8000, 9000],
    by_epoch=False)


runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False , interval=2000)

evaluation = dict(interval=500, metric='mIoU', pre_eval=True)
# work_dir = "/workspace/DATASET/TIGER-Dataset/wsirois/SEGWEIGHT/fold_1_uppernet-Knet_512_4x-SGD_0.0001"

#runtime config
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True