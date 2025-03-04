import os
import json
import math
import copy
import torch
import argparse
from typing import List, Dict, Optional

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine import MMLogger


##############################################################################
# (A) Custom: Server-side Aggregator
##############################################################################
class UncertaintyFedAggregator:
    """
    Supported aggregation methods:
      - 'fedavg'
      - 'fedadam'
      - 'fedyogi'
      - 'uncertaintyfedavg'
    """
    def __init__(self,
                 server_opt: str = 'fedavg',
                 lr: float = 1.0,
                 beta1: float = 0.9,
                 beta2: float = 0.99,
                 tau: float = 1e-3,
                 alpha=2.0):
        self.server_opt = server_opt.lower()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.alpha = alpha
        # For FedAdam / FedYogi
        self.m_t = None
        self.v_t = None
    
    def aggregate(self,
                  w_global: Dict[str, torch.Tensor],
                  client_weights_list: List[Dict[str, torch.Tensor]],
                  client_nsamples: List[int],
                  client_uncerts: List[float]) -> Dict[str, torch.Tensor]:
        skip_keys = ['running_mean', 'running_var', 'tracked']

        # Compute weights based on uncertainty and sample size
        total_n = sum(client_nsamples)
        uncerts = torch.tensor(client_uncerts, dtype=torch.float32)
        if uncerts.max() - uncerts.min() < 1e-6:  # Avoid division by zero
            norm_uncert = torch.ones_like(uncerts)
        else:
            norm_uncert = (uncerts - uncerts.min()) / (uncerts.max() - uncerts.min() + 1e-6)
        
        # Reliability: Lower uncertainty -> higher weight (exponential decay)
        reliability = torch.exp(-self.alpha * norm_uncert)  # Hyperparameter 2.0 can be tuned
        sample_weights = torch.tensor([math.sqrt(n / total_n) for n in client_nsamples], 
                                    dtype=torch.float32)
        combined_weights = sample_weights * reliability
        weight_sum = combined_weights.sum()
        if weight_sum < 1e-6:
            weight_factors = [1.0 / len(client_nsamples)] * len(client_nsamples)
        else:
            weight_factors = (combined_weights / weight_sum).tolist()

        # Aggregate weights
        delta = {}
        for k in w_global.keys():
            if w_global[k].dtype.is_floating_point and not any(s in k for s in skip_keys):
                delta[k] = torch.zeros_like(w_global[k])
                for i, w_i in enumerate(client_weights_list):
                    delta[k] += (w_global[k] - w_i[k]) * weight_factors[i]

        # Update global weights
        if self.server_opt in ['fedavg', 'uncertaintyfedavg']:
            for k in delta:
                w_global[k] -= self.lr * delta[k]

        return w_global

##############################################################################
# (B) Utility Functions: Load/Save (Only Load Model Weights)
##############################################################################

def load_model_weights(runner, ckpt_path):
    """Load only the state_dict into runner.model, without loading optimizer/scheduler states."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    runner.model.load_state_dict(state_dict, strict=False)
    print("Finished loading weights")

def save_full_checkpoint(runner, checkpoint_path):
    """Save the full model checkpoint including optimizer states."""
    out_dir = os.path.dirname(checkpoint_path)
    filename = os.path.basename(checkpoint_path)
    runner.save_checkpoint(out_dir=out_dir, filename=filename)

def count_samples(coco_json_path: str):
    """Count the number of images in a COCO dataset JSON file."""
    if not coco_json_path or not os.path.isfile(coco_json_path):
        return 0
    with open(coco_json_path,'r') as f:
        data = json.load(f)
    return len(data['images'])

##############################################################################
# (C) Key: Auto-Scaling Original YOLOX 300-Epoch Scheduler
##############################################################################
def adapt_param_scheduler_for_local_epoch(cfg, local_epoch):
    """
      1) QuadraticWarmupLR (5 epochs)
      2) CosineAnnealingLR (up to 285 epochs)
      3) ConstantLR (last 15 epochs)
    This is scaled using scale_factor = local_epoch / 300.
    """
    # Find param_scheduler
    if not hasattr(cfg, 'param_scheduler') or not cfg.param_scheduler:
        return
    param_schedulers = cfg.param_scheduler

    scale_factor = local_epoch / 300.0

    # Original (300 epochs) segmentation: [0,5] warmup -> [5,285] cosine -> [285,300] constant
    warmup_end = max(1, int(round(5 * scale_factor)))  # Ensure warmup is at least 1
    last_epochs = int(round(15 * scale_factor))
    cos_start = warmup_end
    
    # Ensure Cosine phase is at least 1 epoch and does not exceed local_epoch
    cos_end = min(local_epoch - 1 if last_epochs == 0 else local_epoch - last_epochs, 
                  max(cos_start + 1, local_epoch - last_epochs))

    # 1) Warmup
    if len(param_schedulers) >= 1:
        sch1 = param_schedulers[0]
        if hasattr(sch1, 'type') and 'Warmup' in sch1.type:
            sch1.begin = 0
            sch1.end = warmup_end
            sch1.convert_to_iter_based = True

    # 2) Cosine Annealing
    if len(param_schedulers) >= 2:
        sch2 = param_schedulers[1]
        sch2.begin = cos_start
        sch2.end = cos_end
        sch2.T_max = cos_end - cos_start  # Key adjustment
        sch2.convert_to_iter_based = True

    # 3) Constant Learning Rate
    if len(param_schedulers) >= 3:
        sch3 = param_schedulers[2]
        sch3.begin = cos_end
        sch3.end = local_epoch
        sch3.by_epoch = True  # Keep LR constant

    # Update max_epochs in train_cfg
    if hasattr(cfg, 'train_cfg'):
        cfg.train_cfg['max_epochs'] = local_epoch

##############################################################################
# (D) Local Training: Load Model Weights Without Loading Optimizer -> Train Directly for local_epoch
##############################################################################
def local_train(config_path: str,
                train_ann_file: str,
                val_ann_file: str,
                work_dir: str,
                local_epoch: int,
                batch_size: int,
                lr: float,
                init_ckpt: str = None,
                freeze_bn: bool = False):
    """
    - Does not use partial_epoch anymore
    - Trains directly for local_epoch
    - Loads only model parameters from init_ckpt (does not resume training state)
    - Automatically scales YOLOX 300-epoch schedule -> local_epoch
    """
    
    # Use MMEngine global logger
    logger = MMLogger.get_current_instance()

    cfg = Config.fromfile(config_path, lazy_import=False)
    cfg.work_dir = work_dir

    # 1) Remove EMAHook
    if 'custom_hooks' in cfg:
        new_hooks = []
        for hook in cfg.custom_hooks:
            if hook.get('type', '') == 'EMAHook':
                continue
            new_hooks.append(hook)
        cfg.custom_hooks = new_hooks

    # 2) Set dataset
    if train_ann_file:
        if 'dataset' in cfg.train_dataloader.dataset:
            cfg.train_dataloader.dataset['dataset']['ann_file'] = train_ann_file
        else:
            cfg.train_dataloader.dataset.ann_file = train_ann_file
    if val_ann_file:
        cfg.val_dataloader.dataset.ann_file = val_ann_file

    # 3) Scale param_scheduler (for YOLOX 300-epoch)
    adapt_param_scheduler_for_local_epoch(cfg, local_epoch)

    # 4) Batch size & learning rate
    if batch_size is not None:
        cfg.train_dataloader.batch_size = batch_size
    if lr is not None:
        if not hasattr(cfg, 'optim_wrapper'):
            cfg.optim_wrapper = dict(optimizer=dict(lr=lr))
        else:
            if 'optimizer' not in cfg.optim_wrapper:
                cfg.optim_wrapper['optimizer'] = dict(lr=lr)
            else:
                cfg.optim_wrapper['optimizer']['lr'] = lr

    # 5) Build runner
    runner = Runner.from_cfg(cfg)
    
    # Reset global_step (ensures annealing resets at each round)
    if hasattr(runner.model.bbox_head, 'global_step'):
        runner.model.bbox_head.global_step.zero_()

    # Dynamically adjust annealing_step based on local_epoch proportion
    if hasattr(runner.model.bbox_head, 'annealing_step'):
        base_annealing = 1000  # Base annealing_step for 20 epochs
        runner.model.bbox_head.annealing_step = int(base_annealing * (local_epoch / 20))
        logger.info(f"[local_train] => Adjusted annealing_step: {runner.model.bbox_head.annealing_step}")

    # 6) If init_ckpt exists => Load only state_dict
    if init_ckpt is not None and os.path.isfile(init_ckpt):
        logger.info(f"[local_train] init_ckpt exists: {init_ckpt}. Now loading weights...")
        state_dict = torch.load(init_ckpt, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        runner.model.load_state_dict(state_dict, strict=False)
        logger.info("[local_train] => Finished loading init_ckpt.")
    else:
        logger.info("[local_train] => init_ckpt is None or not found. Using YOLOX default initialization.")

    # Do not resume -> new optimizer and scheduler states
    runner.train_loop._epoch = 0
    runner.train_loop._max_epochs = local_epoch

    # 7) Optionally freeze batch normalization layers
    if freeze_bn:
        for _, mod in runner.model.named_modules():
            if hasattr(mod, 'track_running_stats'):
                mod.track_running_stats = False
                mod.eval()
        logger.info("[local_train] => Frozen Batch Normalization statistics")

    # 8) Start training
    runner.train()

    # 9) Save complete checkpoint (includes new optimizer state, but won’t be used next round)
    final_ckpt_path = os.path.join(work_dir, 'local_final_ckpt.pth')
    save_full_checkpoint(runner, final_ckpt_path)

    # 10) Extract final weights
    final_weights = copy.deepcopy(runner.model.state_dict())
    del runner
    torch.cuda.empty_cache()
    return final_weights, final_ckpt_path


##############################################################################
# (E) Evaluation Function
##############################################################################
def evaluate_model(config_path: str, work_dir: str, weights: dict):
    """Evaluate model performance using the given weights."""
    cfg = Config.fromfile(config_path, lazy_import=False)
    cfg.work_dir = work_dir
    runner = Runner.from_cfg(cfg)
    runner.model.load_state_dict(weights, strict=False)
    metrics = runner.val()
    del runner
    torch.cuda.empty_cache()
    return metrics


##############################################################################
# (F) Federated Training Process: Each Round Trains for local_epoch
##############################################################################
class FederatedRunner:
    def __init__(self,
                 config_path: str,
                 client_train_jsons: list,
                 val_json: str,
                 server_opt: str,
                 server_lr: float,
                 fed_rounds: int,
                 client_num: int,
                 local_epoch: int,
                 batch_size: int,
                 lr: float,
                 alpha: float,
                 init_model_ckpt: str,
                 fed_work_dir: str,
                 freeze_bn: bool = False,
                 use_uncertainty: bool = False):
        """
        1) If init_model_ckpt is valid, load it into self.global_weights; otherwise, set self.global_weights=None.
        2) If global_weights is not None, save it as global_init.pth and distribute it to all clients.
        """
        self.logger = MMLogger.get_instance("FederatedRunner")
        self.logger.info("[FederatedRunner.__init__] => Initializing...")

        self.config_path = config_path
        self.client_train_jsons = client_train_jsons
        self.val_json = val_json
        self.server_opt = server_opt
        self.server_lr = server_lr
        self.fed_rounds = fed_rounds
        self.client_num = client_num
        self.local_epoch = local_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.init_model_ckpt = init_model_ckpt
        self.fed_work_dir = fed_work_dir
        self.freeze_bn = freeze_bn
        self.use_uncertainty = use_uncertainty

        os.makedirs(self.fed_work_dir, exist_ok=True)

        # 1) Create server-side aggregator
        self.logger.info(f"[FederatedRunner] => Creating aggregator with server_opt={server_opt}, lr={server_lr}")
        self.aggregator = UncertaintyFedAggregator(server_opt=self.server_opt,
                                                   lr=self.server_lr,
                                                   alpha=alpha)

        # 2) If init_model_ckpt exists, load it
        if init_model_ckpt and os.path.isfile(init_model_ckpt):
            c_ = torch.load(init_model_ckpt, map_location='cpu')
            self.global_weights = c_['state_dict'] if 'state_dict' in c_ else c_
            self.logger.info(f"[FederatedRunner] => Loaded init_model_ckpt: {init_model_ckpt}")
        else:
            self.logger.info("[FederatedRunner] => No init_model_ckpt found. Using YOLOX default initialization.")
            self.global_weights = None

        # 3) Initialize clients
        self.clients = []
        for i in range(client_num):
            fname = client_train_jsons[i] if i < len(client_train_jsons) else None
            self.clients.append({
                'name': f'client_{i+1}',
                'train_ann_file': fname,
                'val_ann_file': val_json,
                'local_ckpt': None  # Will be updated based on global_weights
            })

        # 4) If initial global_weights exist, save and distribute
        if self.global_weights is not None:
            init_global_ckpt = os.path.join(self.fed_work_dir, "global_init.pth")
            torch.save({'state_dict': self.global_weights}, init_global_ckpt)
            for i in range(self.client_num):
                self.clients[i]['local_ckpt'] = init_global_ckpt
            self.logger.info(f"[FederatedRunner] => Saved initial global ckpt -> {init_global_ckpt}")

        self.logger.info("[FederatedRunner] => Initialization complete.")

    def run(self):
        """
        Each round consists of:
          (A) Clients load local_ckpt -> perform local training
          (B) Server aggregates results -> updates self.global_weights
          (C) Save global_ckpt -> global_roundX.pth
          (D) Evaluate & distribute to clients for the next round
        """

        logger = MMLogger.get_instance("FederatedRunner")

        for round_idx in range(1, self.fed_rounds + 1):
            logger.info(f"\n===== [Round {round_idx}] Local Training =====")
            client_weights_list = []
            client_nsamples_list = []
            client_uncerts_list = []

            # === (A) Train each client sequentially ===
            for i, cinfo in enumerate(self.clients):
                c_work_dir = os.path.join(self.fed_work_dir, f"{cinfo['name']}_round{round_idx}")
                os.makedirs(c_work_dir, exist_ok=True)

                init_ckpt = cinfo['local_ckpt']
                logger.info(f"[Round {round_idx}] => Client {cinfo['name']} uses init_ckpt={init_ckpt}")

                # Perform local training
                final_weights, new_local_ckpt = local_train(
                    config_path=self.config_path,
                    train_ann_file=cinfo['train_ann_file'],
                    val_ann_file=cinfo['val_ann_file'],
                    work_dir=c_work_dir,
                    local_epoch=self.local_epoch,
                    batch_size=self.batch_size,
                    lr=self.lr,
                    init_ckpt=init_ckpt,
                    freeze_bn=self.freeze_bn
                )

                # Collect trained weights
                client_weights_list.append(final_weights)
                # Count the number of samples in each client
                n_samp = count_samples(cinfo['train_ann_file'])
                client_nsamples_list.append(n_samp)

                # Compute uncertainty (optional)
                if self.use_uncertainty:
                    unc_ = compute_uncertainty(
                        config_path=self.config_path,
                        work_dir=c_work_dir,
                        weights=final_weights,
                        sample_json=cinfo['val_ann_file'],
                        max_samples=30
                    )
                    client_uncerts_list.append(unc_)
                    logger.info(f"[Client {cinfo['name']}] => #Samples={n_samp}, Uncertainty={unc_:.4f}")
                else:
                    client_uncerts_list.append(0.0)

                # Set the latest local checkpoint for this client
                self.clients[i]['local_ckpt'] = new_local_ckpt

            # === (B) Server-side Aggregation ===
            logger.info(f"\n===== [Round {round_idx}] Server Aggregation =====")
            if self.global_weights is None:
                # If no global model yet (first round), choose the client with the lowest uncertainty
                min_uncert_index = client_uncerts_list.index(min(client_uncerts_list))
                self.global_weights = copy.deepcopy(client_weights_list[min_uncert_index])

            # Perform model aggregation
            self.global_weights = self.aggregator.aggregate(
                w_global=self.global_weights,
                client_weights_list=client_weights_list,
                client_nsamples=client_nsamples_list,
                client_uncerts=client_uncerts_list
            )

            # === (C) Save updated global model checkpoint ===
            global_ckpt = os.path.join(self.fed_work_dir, f"global_round{round_idx}.pth")
            torch.save({'state_dict': self.global_weights}, global_ckpt)
            logger.info(f"[Round {round_idx}] => Saved updated global ckpt => {global_ckpt}")

            # === (D) Evaluate the global model & distribute to clients for the next round ===
            eval_dir = os.path.join(self.fed_work_dir, f"eval_round{round_idx}")
            os.makedirs(eval_dir, exist_ok=True)
            metrics = evaluate_model(self.config_path, eval_dir, self.global_weights)
            logger.info(f"[Round {round_idx}] => Global evaluation metrics: {metrics}")

            # Distribute updated global model to all clients
            for i in range(self.client_num):
                self.clients[i]['local_ckpt'] = global_ckpt

        logger.info("[FederatedRunner.run] => ALL ROUNDS COMPLETED.\n")


##############################################################################
# (G) Command-Line Interface
##############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Evidential YOLOX + Federated Learning (Local Epoch version)")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YOLOX 300-epoch config.')
    parser.add_argument('--client_train_jsons', type=str, nargs='+', required=True,
                        help='Training dataset JSON files for each client.')
    parser.add_argument('--val_json', type=str, required=True,
                        help='Validation dataset JSON file.')
    parser.add_argument('--server_opt', type=str, default='fedavg',
                        help='Server aggregation method [fedavg, fedadam, fedyogi, uncertaintyfedavg].')
    parser.add_argument('--server_lr', type=float, default=1.0)
    parser.add_argument('--federated_rounds', type=int, default=5,
                        help='Number of federated learning rounds.')
    parser.add_argument('--client_num', type=int, default=2,
                        help='Number of clients participating in training.')
    parser.add_argument('--local_epoch', type=int, default=20,
                        help='Number of epochs for local training in each round.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for local training.')
    parser.add_argument('--alpha', type=float, default=2.0,
                        help='Hyperparameter for uncertainty weighting in aggregation.')
    parser.add_argument('--init_model_ckpt', type=str, default=None,
                        help='Path to initial model checkpoint.')
    parser.add_argument('--work_dir', type=str, default='./work_dirs/fed_evidential',
                        help='Directory to save federated learning results.')
    parser.add_argument('--freeze_bn', action='store_true',
                        help='Freeze batch normalization layers during training.')
    parser.add_argument('--use_uncertainty', action='store_true',
                        help='Use uncertainty weighting in model aggregation.')

    return parser.parse_args()


def main():
    """Main function to start federated learning."""
    args = parse_args()
    fed_runner = FederatedRunner(
        config_path=args.config,
        client_train_jsons=args.client_train_jsons,
        val_json=args.val_json,
        server_opt=args.server_opt,
        server_lr=args.server_lr,
        fed_rounds=args.federated_rounds,
        client_num=args.client_num,
        local_epoch=args.local_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        init_model_ckpt=args.init_model_ckpt,
        fed_work_dir=args.work_dir,
        freeze_bn=args.freeze_bn,
        use_uncertainty=args.use_uncertainty
    )
    fed_runner.run()


if __name__ == '__main__':
    main()

