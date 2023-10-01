# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = './mmclassification/checkpoints/resnext101_32x8d_b32x8_imagenet_20210506-23a247d5.pth'
load_from = './models/resnest101_imagenet_converted-032caa52.pth'
resume_from = None
workflow = [('train', 1)]
