import os
import yaml
import torch

def check_gpu_status():
    """
    Check GPU availability and print status
    """
    if torch.cuda.is_available():
        print("\nGPU Status:")
        print(f"- CUDA Available: Yes")
        print(f"- GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"- GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return True
    else:
        print("\nNo CUDA-capable GPU detected. Training will run on CPU (not recommended).")
        return False

def create_dataset_yaml(dataset_path, output_path):
    """
    Create YAML configuration file for YOLOv5 training
    """
    dataset_yaml = {
        'path': dataset_path,
        'train': 'train',
        'val': 'valid',
        'test': 'test',
        'nc': 1,
        'names': ['license_plate']
    }

    with open(output_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"Created dataset configuration at: {output_path}")

def train_yolov5(
    dataset_yaml_path,
    img_size=640,
    batch_size=16,
    epochs=100,
    model_size='s',
    workers=8,
    device='0'  # GPU device ID, or 'cpu'
):
    """
    Train YOLOv5 model with CUDA support, without wandb logging
    """
    # Check GPU status
    has_cuda = check_gpu_status()
    if not has_cuda and device != 'cpu':
        print("WARNING: CUDA not available. Forcing CPU usage.")
        device = 'cpu'
    
    # Clone YOLOv5 repository if not exists
    if not os.path.exists('yolov5'):
        print("\nCloning YOLOv5 repository...")
        os.system('git clone https://github.com/ultralytics/yolov5.git')
    
    # Install dependencies
    print("\nInstalling dependencies...")
    os.system('pip install -r yolov5/requirements.txt')
    
    # Calculate optimal batch size based on GPU memory if using CUDA
    if has_cuda:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        if gpu_mem < 8:  # Less than 8GB
            batch_size = min(batch_size, 8)
            print(f"\nAdjusting batch size to {batch_size} due to GPU memory constraints")
    
     # Set environment variables to disable wandb
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['WANDB_DISABLED'] = 'true'

    # Training command with explicit wandb disable flags
    train_command = (
        f"python yolov5/train.py "
        f"--img {img_size} "
        f"--batch {batch_size} "
        f"--epochs {epochs} "
        f"--data {dataset_yaml_path} "
        f"--weights yolov5{model_size}.pt "
        f"--workers {workers} "
        f"--device {device} "
        f"--cache "
        f"--project license_plate_detection "
        f"--name yolov5{model_size}_run1 "
        f"--exist-ok"
    )
     
    
    print("\nStarting training with following configuration:")
    print(f"- Image size: {img_size}")
    print(f"- Batch size: {batch_size}")
    print(f"- Epochs: {epochs}")
    print(f"- Model: YOLOv5{model_size}")
    print(f"- Device: {'CUDA' if device != 'cpu' else 'CPU'}")
    print(f"- Workers: {workers}")
    print(f"- Logging: Local (wandb disabled)")
    
    os.system(train_command)

if __name__ == "__main__":
    # Example configuration
    config = {
        'dataset_path': 'c:/data/input',
        'yaml_path': 'dataset.yaml'
    }
    
    # Create dataset YAML
    create_dataset_yaml(config['dataset_path'], config['yaml_path'])
    
    # Start training with CUDA if available, without wandb
    train_yolov5(
        dataset_yaml_path=config['yaml_path'],
        img_size=640,
        batch_size=16,        # Will be automatically adjusted based on GPU memory
        epochs=100,
        model_size='s',
        workers=8,
        device='0'           # Use '0' for first GPU, '1' for second GPU, 'cpu' for CPU
    )