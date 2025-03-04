import os

##############################################################################
# 1) base config
##############################################################################
_base_ = [
    '../configs/yolox/yolox_s_8xb8-300e_coco.py'  # YOLOX-s congig from mmdet
]

##############################################################################
# 2) 
##############################################################################

dataset_type = 'CocoDataset'
data_root = '../datasets/KITTI/'
classes = ("Car", "Pedestrian", "Cyclist", "Truck",
           "Person_sitting", "Tram", "Misc")

backend_args = None  

##############################################################################
# 3) 
##############################################################################
model = dict(
    bbox_head=dict(num_classes=len(classes))
)

##############################################################################
# 4) 
##############################################################################

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/path/to/your/coco_train.json',  
        data_prefix=dict(img='/path/to/your/training_data'),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=backend_args,
        metainfo=dict(classes=classes)
    ),
    pipeline={{_base_.train_pipeline}}  
)

train_dataloader = dict(
    _delete_=True,  
    batch_size=8,   
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset
)

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    _delete_=True, 
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/path/to/your/coco_val.json',  
        data_prefix=dict(img='/path/to/your/val_data'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
        metainfo=dict(classes=classes)
    )
)
test_dataloader = val_dataloader

##############################################################################
# 5) Evaluator
##############################################################################
val_evaluator = dict(
    _delete_=True, 
    type='CocoMetric',
    ann_file='/path/to/your/coco_val.json',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

##############################################################################
# 6) 
##############################################################################
work_dir = './work_dirs/kitti_yolox_s'
if not os.path.exists(work_dir):
    os.makedirs(work_dir, exist_ok=True)

max_epochs=300
train_cfg = dict(max_epochs=max_epochs, val_interval=10)

# only for fed training
custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=max_epochs // 20,  
        priority=48),
    dict(
        type='SyncNormHook', 
        priority=48),
    dict(
        type='EMAHook',  
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]

default_hooks = dict(
    checkpoint=dict(
        interval=10,
        max_keep_ckpts=3,
        save_best=None
    )
)

print("kitti_yolox.py created!")
