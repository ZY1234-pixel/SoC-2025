# InternImage-L + UPerHead inference config for binary document segmentation.

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='InternImage',
        core_op='DCNv3',
        channels=160,
        depths=[5, 5, 22, 5],
        groups=[10, 20, 40, 80],
        mlp_ratio=4.,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=2.0,
        post_norm=True,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        init_cfg=None),
    decode_head=dict(
        type='UPerHead',
        in_channels=[160, 320, 640, 1280],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='DocSegCombinedLoss',
            bce_weight=1.0,
            dice_weight=1.0,
            boundary_weight=0.5,
            boundary_start_epoch=6,
            boundary_theta0=3.0,
            dice_smooth=1.0,
            loss_name='loss_docseg')),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=640,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='DocSegCombinedLoss',
            bce_weight=0.4,
            dice_weight=0.4,
            boundary_weight=0.0,
            boundary_start_epoch=999999,
            dice_smooth=1.0,
            loss_name='loss_aux_docseg')),
    train_cfg=dict(),
    test_cfg=dict(mode='whole', threshold=0.60, keep_input_size=True))

test_cfg = dict(mode='whole', threshold=0.60, keep_input_size=True)
