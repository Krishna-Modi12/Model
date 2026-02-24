import sys, os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.config import get_config_dict
from src.models.face_analysis_model import FaceAnalysisModel
from src.training.trainer import FaceAnalysisLightningModule

def main():
    print("="*50)
    print("SANITY CHECK: Unfreeze & Scheduler")
    print("="*50)

    config = get_config_dict()
    
    # 1. Initialize Lightning Module (which builds the model and optimizers)
    model_module = FaceAnalysisLightningModule(config)
    
    # 2. Check Optimizer Param Groups
    opt_dicts = model_module.configure_optimizers()
    optimizer = opt_dicts["optimizer"]
    scheduler_dict = opt_dicts["lr_scheduler"]["scheduler"]
    
    print("\n--- Optimizer Configurations ---")
    for i, group in enumerate(optimizer.param_groups):
        print(f"Group {i} '{group.get('name', 'N/A')}': LR = {group['lr']}")
        
    print("\n--- Scheduler Audit ---")
    if hasattr(scheduler_dict, '_schedulers'):
        warmup_sched = scheduler_dict._schedulers[0]
        cosine_sched = scheduler_dict._schedulers[1]
        print(f"SequentialLR Milestones: {scheduler_dict._milestones}")
        print(f"Warmup Total Iters: {warmup_sched.total_iters}")
        print(f"CosineAnnealingLR T_max: {cosine_sched.T_max}")
    else:
        print("Scheduler is not SequentialLR.")

    print("\n--- Scheduler Dynamics Simulation ---")
    print(f"steps_per_epoch: {config['scheduler']['steps_per_epoch']}")
    print(f"warmup_steps: {config['scheduler']['warmup_steps']}")
    print(f"t_max_steps: {config['scheduler']['t_max_steps']}")
    
    warmup_s = config['scheduler']['warmup_steps']
    total_s = warmup_s + config['scheduler']['t_max_steps']
    
    steps_to_log = [1, 50, warmup_s, warmup_s + 1, warmup_s + 1000, warmup_s + 4000, total_s]
    print(f"\nStep | Backbone LR | Head LR")
    print("-" * 35)
    
    for step in range(1, total_s + 2):
        optimizer.step()
        scheduler_dict.step()
        
        if step in steps_to_log:
            backbone_lr = optimizer.param_groups[0]['lr']
            head_lr = optimizer.param_groups[1]['lr']
            print(f"{step:<4} | {backbone_lr:.2e}  | {head_lr:.2e}")

    # 3. Simulate Unfreeze
    print("\n--- Parameter Trainability Audit (Simulating Epoch 5) ---")
    # By default, config has freeze_backbone=True in FaceAnalysisLightningModule
    total_frozen_initial = sum(1 for p in model_module.model.backbone.parameters() if not p.requires_grad)
    total_trainable_initial = sum(1 for p in model_module.parameters() if p.requires_grad)
    print(f"Before Epoch 5 - Frozen backbone params: {total_frozen_initial}")
    print(f"Before Epoch 5 - Total trainable model params: {total_trainable_initial}")

    # Unfreeze fully just like the training loop does
    model_module.model.unfreeze_backbone(num_blocks=None)
    
    total_frozen_after = sum(1 for p in model_module.model.backbone.parameters() if not p.requires_grad)
    total_trainable_after = sum(1 for p in model_module.parameters() if p.requires_grad)
    print(f"\nAfter Epoch 5 - Frozen backbone params: {total_frozen_after}")
    print(f"After Epoch 5 - Total trainable model params: {total_trainable_after}")

    print("\n--- Backbone Stem Layer Trainability ---")
    # Identify early stem layers
    # Print the first 10 named parameters in the backbone to confirm early layers are unfrozen
    backbone_params = list(model_module.model.backbone.named_parameters())
    for name, param in backbone_params[:10]:
        print(f"Layer: {name:<30} | requires_grad: {param.requires_grad}")
    
    print("\n... and checking the last 3 layers ...")
    for name, param in backbone_params[-3:]:
        print(f"Layer: {name:<30} | requires_grad: {param.requires_grad}")

if __name__ == "__main__":
    main()
