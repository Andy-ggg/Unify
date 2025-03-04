import os

# 1. base config
_base_ = [
    '../configs/yolox/yolox_s_8xb8-300e_coco.py'
]

dataset_type = 'CocoDataset'
data_root = '../datasets/nuimages/'
classes = (
    "vehicle.bicycle",
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.car",
    "vehicle.construction",
    "vehicle.ego",
    "vehicle.emergency.ambulance",
    "vehicle.emergency.police",
    "vehicle.motorcycle",
    "vehicle.trailer",
    "vehicle.truck",
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer",
    "human.pedestrian.stroller",
    "human.pedestrian.wheelchair",
    "movable_object.barrier",
    "movable_object.debris",
    "movable_object.pushable_pullable",
    "movable_object.trafficcone",
)

backend_args = None  

# 2. custom module
custom_imports = dict(
    imports=['src.yolox_evidential_head'],
    allow_failed_imports=False)


# custom head: EvidentialYOLOXHead
model = dict(
    type='YOLOX',
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],  
        out_channels=128,             
        num_csp_blocks=1
    ),
    bbox_head=dict(
        type='EvidentialYOLOXRegHead',
        num_classes=len(classes),
        kl_beta=0.,
        #  Original parameters of YOLOXHead:
        in_channels=128,   
        feat_channels=128, 
        stacked_convs=2,
        strides=(8, 16, 32),
        # Hyperparameters of EL
        annealing_step=1000,
        reg_evd_loss_weight=0.01,
        loss_cls=dict(  
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', reduction='sum', loss_weight=5.0),  
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
    )
)


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
# 6) other configs
##############################################################################
work_dir = './work_dirs/kitti_yolox_s_evd'
if not os.path.exists(work_dir):
    os.makedirs(work_dir, exist_ok=True)

max_epochs=300
train_cfg = dict(max_epochs=max_epochs, val_interval=10)

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
        priority=49),

]

default_hooks = dict(
    checkpoint=dict(
        interval=10,
        max_keep_ckpts=3,
        save_best=None
    )
)

print("nuimage_evd.py created!")
