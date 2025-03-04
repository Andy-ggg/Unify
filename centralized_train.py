import argparse
from mmengine.config import Config
from mmengine.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='custom_configs/kitti_yolox.py',
                        help='Path to the config file')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Local learning rate. If None, use config default.')
    return parser.parse_args()

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config, lazy_import=False)

    if args.epochs is not None:
        cfg.train_cfg['max_epochs'] = args.epochs

    if args.batch_size is not None:
        cfg.train_dataloader['batch_size'] = args.batch_size
        
    if args.lr is not None:
        if 'optim_wrapper' not in cfg:
            cfg.optim_wrapper = dict(optimizer=dict(lr=args.lr))
        else:
            if 'optimizer' not in cfg.optim_wrapper:
                cfg.optim_wrapper['optimizer'] = dict(lr=args.lr)
            else:
                cfg.optim_wrapper['optimizer']['lr'] = args.lr
                

    runner = Runner.from_cfg(cfg)
    # print(runner.model)
    runner.train()

if __name__ == '__main__':
    main()
