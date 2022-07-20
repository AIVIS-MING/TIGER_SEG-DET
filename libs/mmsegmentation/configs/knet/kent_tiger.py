#_base_ = [ '../_base_/datasets/tiger.py']
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
num_stages = 3
conv_kernel_size = 1
_num_classes = 7
#size 4 ,8 ,16(max) per gpu
_batch_size = 4
_base_lr = 0.0001

model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='IterativeDecodeHead',
        num_stages=num_stages,
        kernel_update_head=[
            dict(
                type='KernelUpdateHead',
                num_classes=_num_classes,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=512,
                out_channels=512,
                dropout=0.0,
                conv_kernel_size=conv_kernel_size,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN'))) for _ in range(num_stages)
        ],
        kernel_generate_head=dict(
            type='UPerHead',
            in_channels=[256, 512, 1024, 2048],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=_num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
            # loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            #              dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]
            )),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=_num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
        # loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        #              dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(256, 256))
    #test_cfg=dict(mode='whole')
    )


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile_Tiger'),
    dict(type='LoadAnnotations__Tiger', reduce_zero_label=True),
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
        img_scale=(2048, 2048),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        #dict(type='Resize', keep_ratio=True),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio = True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'TIGERDataset'
data_root = '/workspace/DATASET/TIGER-Dataset/wsirois/Seg_new4'
data = dict(
    samples_per_gpu=_batch_size,
    workers_per_gpu=4,
    reduce_zero_label = True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train_1_fold_512_4x.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val_1_fold_512_4x.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val_1_fold_512_4x.txt',
        pipeline=test_pipeline))

# optimizer Setting==========================
# optimizer = dict(type='SGD', lr=_base_lr, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

optimizer = dict(type='AdamW', lr=_base_lr, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[3000],
    by_epoch=False)


runner = dict(type='IterBasedRunner', max_iters=6000)
checkpoint_config = dict(by_epoch=False , interval=6000)
evaluation = dict(interval=200, metric='mIoU', pre_eval=True, reduce_zero_label=True)
work_dir = "/workspace/DATASET/TIGER-Dataset/wsirois/SEGWEIGHT/uppernet-Knet_512_r50_bs4_seg4_random_crop"
#runtime config
log_config = dict(
    interval=20,
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

# bs/gpu=16  bs_total=64  iter(max) 500
# bs/gpu=2   bs_total=8   iter(max) 4000