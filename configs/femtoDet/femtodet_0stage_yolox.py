_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    # './yolox_tta.py'
]
classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

img_scale = (640, 640)  # width, height
test_img_scale = (416, 416)

# model settings
widen_factor = 0.25
pretrain_weight = './weights/femtodet_nano_backbone_pretrained_model_witc_cd_conv2_4973.pth'
load_from = None
resume_from = None
default_channels = [32, 96, 320]
neck_in_chanels = [int(ch * widen_factor) for ch in default_channels]
headfeat_channel = 64

model = dict(
    type='FemtoDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(320, 640),
                size_divisor=32,
                interval=10)
        ]),
    backbone=dict(
        type='FemtoNet',
        widen_factor=widen_factor,
        diff_conv=True,
        out_indices=(2, 4, 6),
        act_cfg=dict(type='ReLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone',
            checkpoint=pretrain_weight)),
    neck=dict(
        type='SharedNeck',
        in_channels=neck_in_chanels,
        out_channels=headfeat_channel,
        fixed_size_idx=1,
        add=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU'),
        num_outs=1),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=len(classes),
        in_channels=headfeat_channel,
        feat_channels=headfeat_channel,
        strides=[16, ],
        stacked_convs=0,
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        act_cfg=dict(type='ReLU'),
        use_depthwise=True),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
data_root = '/media/yzs/ubuntu/datasets/VOC'
dataset_type = 'CocoDataset'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.82, 58.82, 58.82], to_rgb=True)

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0, prob=0.5),
    dict(
        type='RandomAffine',
        superposition=False,
        scaling_ratio_range=(0.5, 1.5),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='FemtoDetMixUp',
        superposition=False,
        img_scale=img_scale,
        ratio_range=(0.8, 1.2),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug_EodVersion', hgain=0.015, sgain=0.7, vgain=0.4),
    dict(type='RandomFlip', prob=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    # Resize and Pad are for the last 15 epochs when Mosaic,
    # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/media/yzs/ubuntu/datasets/VOC/coco/voc0712_trainval.json',
        data_prefix=dict(img='VOCdevkit/'),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=backend_args),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=test_img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/media/yzs/ubuntu/datasets/VOC/coco/voc07_test.json',
        data_prefix=dict(img='VOCdevkit'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    # ann_file=data_root + 'annotations/instances_val2017.json',
    ann_file='/media/yzs/ubuntu/datasets/VOC/coco/voc07_test.json',
    metric='bbox',
    backend_args=backend_args)

test_evaluator = val_evaluator

# training settings
max_epochs = 300
num_last_epochs = 0
interval = 10

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

# optimizer
# default 8 gpu
base_lr = 0.01
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# learning rate
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    # dict(
    #     # use fixed lr during last 15 epochs
    #     type='ConstantLR',
    #     by_epoch=True,
    #     factor=1,
    #     begin=max_epochs - num_last_epochs,
    #     end=max_epochs,
    # )
]

default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))

custom_hooks = [
    dict(
        type='FemtoDetModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48,
        all_unfreeze_after_this_epoch=50
    ),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]
find_unused_parameters = True
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
# auto_scale_lr = dict(base_batch_size=64)
