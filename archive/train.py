import argparse
import mlflow.pytorch
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from loguru import logger

from src.data.dataset import create_dataloaders
from src.training.trainer import FaceAnalysisLightningModule, build_trainer
from src.config import get_config_dict, FACE_SHAPES

def main(args):
    # CRITICAL POWER MITIGATION: Strictly throttle CPU threads to prevent PSU Over Current Protection trips
    # This prevents the CPU from maxing out simultaneously with the GPU when batching data
    torch.set_num_threads(4) 
    
    # Enable deterministic mode to smooth out GPU power draw spikes
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    assert torch.cuda.is_available(), "CUDA not available — training would run on CPU. Aborting!"
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    config = get_config_dict()
    
    # Override batch size if provided in args
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs:
        config["training"]["epochs"] = args.epochs

    logger.info("Initializing DataLoaders...")
    
    if args.resume:
        logger.info("Resume flag detected. Adjusting configuration for FINE-TUNING mode.")
        import os
        config["paths"]["checkpoints"] = os.path.join(config["paths"]["checkpoints"], "finetuned")
        os.makedirs(config["paths"]["checkpoints"], exist_ok=True)
        # Drop learning rate to 1/10th of the original for fine-tuning
        config["optimizer"]["lr"] = config["optimizer"]["lr"] / 10.0
        
        # Extend total maximum epochs slightly (adding 75 epochs to current)
        config["training"]["epochs"] = 175
        
    loaders = create_dataloaders(config)

    logger.info("Computing balanced class weights from training set...")
    train_labels = np.array([
        ann["shape_label"]
        for ann in loaders["train"].dataset.annotations
    ])
    weights = compute_class_weight('balanced',
        classes=np.arange(len(FACE_SHAPES)),
        y=train_labels)
    
    # Store weights in config as a tensor so the LightningModule can use them
    config["training"]["class_weights"] = torch.tensor(weights, dtype=torch.float32)

    logger.info("Building Model and LightningModule...")
    model = FaceAnalysisLightningModule(config)

    logger.info("Starting Training Pipeline...")
    trainer = build_trainer(config)
    
    if args.test_only:
        logger.info("Running explicitly in TEST ONLY mode.")
        if args.resume:
            trainer.test(model, dataloaders=loaders["test"], ckpt_path=args.resume)
        else:
            logger.error("You MUST provide --resume with --test_only")
        return

    # MLflow auto-logging with Lightning
    mlflow.set_experiment("Face Analysis AI")
    mlflow.pytorch.autolog()

    with mlflow.start_run():
        trainer.fit(
            model,
            train_dataloaders=loaders["train"],
            val_dataloaders=loaders["val"],
            ckpt_path=args.resume if args.resume else None
        )
        
        logger.success("Training complete!")
        
        # Test after training using the BEST checkpoint
        trainer.test(model, dataloaders=loaders["test"], ckpt_path="best")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--test_only", action="store_true", help="Only run testing")
    parser.add_argument("--resume", type=str, help="Path to checkpoint")
    args = parser.parse_args()
    
    main(args)
