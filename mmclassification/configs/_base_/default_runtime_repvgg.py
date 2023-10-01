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
load_from = './models/repvgg-B2_3rdparty_4xb64-coslr-120e_in1k_20210909-bd6b937c.pth'
resume_from = None
workflow = [('train', 1)]
