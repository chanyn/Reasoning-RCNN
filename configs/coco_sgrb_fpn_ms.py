# model settings
model = dict(
    type='ThreeStageGraphDetector',
    pretrained='modelzoo://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        use_sigmoid_cls=True),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=[dict(
        type='SharedFCRoIHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=81,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False),
        dict(
            type='ConvFCRoIHeadEnhance',
            enhance_channels=256,
            num_shared_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=81,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=False)
    ],
    graph_convolution=dict(
        latent_graph_channel=256,
        n_kernels_gc=8,
        n_graph_node=512,
        neigh_size=32)
)
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        pos_fraction=0.5,
        pos_balance_sampling=False,
        neg_pos_ub=256,
        allowed_border=0,
        crowd_thr=1.1,
        anchor_batch_size=256,
        pos_iou_thr=0.7,
        neg_iou_thr=0.3,
        neg_balance_thr=0,
        min_pos_iou=0.3,
        pos_weight=-1,
        smoothl1_beta=1 / 9.0,
        debug=False),
    rcnn=dict(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        crowd_thr=1.1,
        roi_batch_size=512,
        add_gt_as_proposals=False,
        pos_fraction=0.25,
        pos_balance_sampling=False,
        neg_pos_ub=512,
        neg_balance_thr=0,
        min_pos_iou=0.5,
        pos_weight=-1,
        debug=False),
    rcnn2=dict(
        pos_iou_thr=0.6,
        neg_iou_thr=0.6,
        crowd_thr=1.1,
        roi_batch_size=512,
        add_gt_as_proposals=False,
        pos_fraction=0.25,
        pos_balance_sampling=False,
        neg_pos_ub=512,
        neg_balance_thr=0,
        min_pos_iou=0.5,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(score_thr=0.001, max_per_img=150, nms_thr=0.55))
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/xuhang/data/detection_data/COCO2017/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        img_scale=[(1333, 600),(1333, 1000)],
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[4])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 5
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/faster_rcnn_r101_fpn_1x'
#load_from = None
#resume_from = './exps/coco_three_stage_graph_fpn_ms/epoch_12.pth'
load_from = './tools/transfer_domian/model/vg_transfer_coco.pth'
resume_from = None
workflow = [('train', 1)]
