# ... (previous imports and warning suppressions remain the same)

def train_yolov7(
    dataset_yaml_path,
    img_size=640,     # Increased to 640
    batch_size=16,    # Increased to utilize free VRAM
    epochs=100,
    workers=8,        # Increased workers
    device='0'
):
    """Train YOLOv7 base model with optimized VRAM usage"""
    start_time = time.time()
    output_dir = 'license_plate_detection'
    os.makedirs(output_dir, exist_ok=True)
    
    # Check GPU status
    has_cuda = check_gpu_status()
    if not has_cuda and device != 'cpu':
        print("WARNING: CUDA not available. Forcing CPU usage.")
        device = 'cpu'
    
    # RTX 3070 specific CUDA optimizations
    if has_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.enabled = True
    
    # Training command optimized for available VRAM
    train_command = (
        f"python -W ignore yolov7/train.py "
        f"--workers {workers} "
        f"--device {device} "
        f"--batch-size {batch_size} "
        f"--data {dataset_yaml_path} "
        f"--img {img_size} {img_size} "
        f"--cfg yolov7/cfg/training/yolov7.yaml "
        f"--weights '' "
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
    print(f"- Device: RTX 3070")
    print(f"- Workers: {workers}")
    print(f"- Model: YOLOv7 (base)")
    print(f"- Output directory: {output_dir}")
    
    print("\nOptimizations enabled:")
    print("- Full resolution (640x640)")
    print("- Increased batch size")
    print("- CUDA optimizations")
    print("- TF32 precision")
    print("- Maximum worker threads")
    
    # Execute training
    print("\nStarting training...")
    os.system(train_command)
    
    # Print completion
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/3600:.2f} hours")

if __name__ == "__main__":
    config = {
        'dataset_path': 'C:/data/input',
        'yaml_path': 'dataset.yaml'
    }
    
    # Create dataset YAML
    create_dataset_yaml(config['dataset_path'], config['yaml_path'])
    
    # Start training with optimized VRAM usage
    train_yolov7(
        dataset_yaml_path=config['yaml_path'],
        img_size=640,     # Full resolution
        batch_size=16,    # Increased batch size
        epochs=10,
        workers=8,        # Maximum workers
        device='0'
    )