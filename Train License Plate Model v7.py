import os
import yaml
import torch
import time
from datetime import datetime, timedelta

# Environment variables to disable wandb
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_SILENT'] = 'true'

def check_gpu_status():
    """Check GPU availability and print detailed status"""
    if torch.cuda.is_available():
        print("\nGPU Status:")
        print(f"- CUDA Available: Yes")
        print(f"- GPU Device: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"- Total GPU Memory: {total_memory:.2f} GB")
        print(f"- Current GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        return True
    else:
        print("\nNo CUDA-capable GPU detected. Training will run on CPU (not recommended).")
        return False

def create_dataset_yaml(dataset_path, output_path):
    """Create YAML configuration file for YOLOv7 training"""
    dataset_yaml = {
        'train': f"{dataset_path}/train",
        'val': f"{dataset_path}/valid",
        'test': f"{dataset_path}/test",
        'nc': 1,
        'names': ['license_plate']
    }

    with open(output_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"Created dataset configuration at: {output_path}")

def train_yolov7(
    dataset_yaml_path,
    img_size=640,
    batch_size=16,
    epochs=300,
    workers=4,
    device='0'
):
    """Train YOLOv7 model"""
    start_time = time.time()
    output_dir = 'license_plate_detection'
    os.makedirs(output_dir, exist_ok=True)
    
    # Check GPU status
    has_cuda = check_gpu_status()
    if not has_cuda and device != 'cpu':
        print("WARNING: CUDA not available. Forcing CPU usage.")
        device = 'cpu'
    
    # Training command
    train_command = (
        f"python yolov7/train.py "
        f"--workers {workers} "
        f"--device {device} "
        f"--batch-size {batch_size} "
        f"--data {dataset_yaml_path} "
        f"--img {img_size} {img_size} "
        f"--cfg yolov7/cfg/training/yolov7.yaml "
        f"--weights '' "  # Training from scratch
        f"--name yolov7_license_plates "
        f"--hyp yolov7/data/hyp.scratch.p5.yaml "
        f"--epochs {epochs} "
        f"--project {output_dir} "
        f"--exist-ok "
    )
    
    # Print configuration
    print("\nTraining Configuration:")
    print(f"- Image size: {img_size}x{img_size}")
    print(f"- Batch size: {batch_size}")
    print(f"- Epochs: {epochs}")
    print(f"- Device: {'CUDA' if device != 'cpu' else 'CPU'}")
    print(f"- Workers: {workers}")
    print(f"- Output directory: {output_dir}")
    
    # Execute training
    print("\nStarting training...")
    os.system(train_command)
    
    # Print completion
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/3600:.2f} hours")

if __name__ == "__main__":
    # Configuration
    config = {
        'dataset_path': 'C:/data/input',  # Update if needed
        'yaml_path': 'dataset.yaml'
    }
    
    # Create dataset YAML
    create_dataset_yaml(config['dataset_path'], config['yaml_path'])
    
    # Start training
    train_yolov7(
        dataset_yaml_path=config['yaml_path'],
        img_size=640,
        batch_size=8,
        epochs=300,
        workers=8,
        device='0'
    )