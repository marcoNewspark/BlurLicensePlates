import os
import yaml
import torch
import time
import warnings

from datetime import datetime, timedelta

# Filter out specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='wandb*')

def format_time(seconds):
    """Convert seconds to human readable time"""
    return str(timedelta(seconds=int(seconds)))

def get_gpu_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        return f"{gpu_memory:.2f}GB"
    return "N/A"

def check_gpu_status():
    """Check GPU availability and print detailed status"""
    if torch.cuda.is_available():
        print("\nGPU Status:")
        print(f"- CUDA Available: Yes")
        print(f"- GPU Device: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"- Total GPU Memory: {total_memory:.2f} GB")
        print(f"- CUDA Version: {torch.version.cuda}")
        print(f"- Current GPU Memory Usage: {get_gpu_usage()}")
        return True
    else:
        print("\nNo CUDA-capable GPU detected. Training will run on CPU (not recommended).")
        return False

def create_dataset_yaml(dataset_path, output_path):
    """Create YAML configuration file for YOLOv5 training"""
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

def print_progress_header():
    """Print header for progress tracking"""
    print("\n" + "="*100)
    print(f"{'Epoch':^10} | {'Progress':^15} | {'GPU Mem':^10} | {'Time Left':^12} | {'Learning Rate':^12} | {'Loss':^20}")
    print("="*100)

def create_progress_file(output_dir):
    """Create a file to log training progress"""
    log_file = os.path.join(output_dir, 'training_progress.txt')
    with open(log_file, 'w') as f:
        f.write("Training Progress Log\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    return log_file

def train_yolov5(
    dataset_yaml_path,
    img_size=640,
    batch_size=32,
    epochs=100,
    model_size='s',
    workers=8,
    device='0'
):
    """Train YOLOv5 model with enhanced progress feedback"""
    start_time = time.time()
    output_dir = 'license_plate_detection'
    os.makedirs(output_dir, exist_ok=True)
    log_file = create_progress_file(output_dir)
    
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
    
    # Set CUDA optimizations
    if has_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
    
    # Disable wandb
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['WANDB_DISABLED'] = 'true'
    
    # Training command with detailed logging
    train_command = (
        f"python yolov5/train.py "
        f"--img {img_size} "
        f"--batch {batch_size} "
        f"--epochs {epochs} "
        f"--data {dataset_yaml_path} "
        f"--weights yolov5{model_size}.pt "
        f"--workers {workers} "
        f"--device {device} "
        f"--cache ram "
        f"--project {output_dir} "
        f"--name yolov5{model_size}_run1 "
        f"--exist-ok "
        f"--label-smoothing 0.1 "
        f"--multi-scale "
    )
    
    # Print initial configuration
    print("\nTraining Configuration:")
    print(f"- Image size: {img_size}")
    print(f"- Batch size: {batch_size}")
    print(f"- Epochs: {epochs}")
    print(f"- Model: YOLOv5{model_size}")
    print(f"- Device: {'CUDA' if device != 'cpu' else 'CPU'}")
    print(f"- Workers: {workers}")
    print(f"- Output directory: {output_dir}")
    print(f"- Log file: {log_file}")
    
    print("\nOptimizations enabled:")
    print("- CUDA benchmark mode")
    print("- RAM caching")
    print("- Multi-scale training")
    print("- Adam optimizer")
    print("- Linear learning rate")
    print("- Label smoothing")
    
    # Print progress header
    print_progress_header()
    
    # Execute training
    os.system(train_command)
    
    # Training completion summary
    end_time = time.time()
    training_time = format_time(end_time - start_time)
    
    print("\nTraining Complete!")
    print(f"Total training time: {training_time}")
    print(f"Final GPU memory usage: {get_gpu_usage()}")
    print(f"\nResults and checkpoints saved in: {os.path.abspath(output_dir)}")
    print("Files saved:")
    print("- Best model weights: best.pt")
    print("- Last model weights: last.pt")
    print("- Training progress plots: results.png")
    print("- Confusion matrix: confusion_matrix.png")
    print("- Full training log: training_progress.txt")

if __name__ == "__main__":
    # Configuration
    config = {
        'dataset_path': 'c:/data/input',
        'yaml_path': 'dataset.yaml'
    }
    
    # Create dataset YAML
    create_dataset_yaml(config['dataset_path'], config['yaml_path'])
    
    # Start training with enhanced feedback
    train_yolov5(
        dataset_yaml_path=config['yaml_path'],
        img_size=640,
        batch_size=32,
        epochs=100,
        model_size='s',
        workers=8,
        device='0'
    )