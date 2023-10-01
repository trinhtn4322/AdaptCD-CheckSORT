# dataset settings
#_base_ = ['./pipelines/auto_aug.py']
dataset_type = 'MyDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='ColorJitter',brightness=0.5,contrast=0.5,saturation=0.5),#色彩抖动
    dict(type='RandomErasing'),#随机擦除
    dict(type='RandomGrayscale',gray_prob=0.5),
    #dict(type='AutoAugment', policies={{_base_.policy_imagenet}}),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,#调整batchsize
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='./data/alladd2/train',#***************
        ann_file='./data/alladd2/meta/train.txt',#****************
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='./data/alladd2/val',#******************
        ann_file='./data/alladd2/meta/val.txt',#***************
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='./data/alladd2/val',#********************
        ann_file='./data/alladd2/meta/val.txt',#*******************
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')