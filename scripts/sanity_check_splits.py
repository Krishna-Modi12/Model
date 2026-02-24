from src.config import get_config_dict
from src.dataset import create_dataloaders
import sys

def main():
    cfg = get_config_dict()
    loaders = create_dataloaders(cfg)
    
    print(f"\nDataLoader Sizes:")
    print(f"Train: {len(loaders['train'].dataset)}")
    print(f"Val:   {len(loaders['val'].dataset)}")
    print(f"Test:  {len(loaders['test'].dataset)}")
    
    try:
        batch = next(iter(loaders["train"]))
        print(f"\nSuccessfully loaded one train batch:")
        print(f"Images shape: {batch['images'].shape}")
        print(f"Labels shape: {batch['shape_labels'].shape}")
    except Exception as e:
        print(f"\nFailed to load batch: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
