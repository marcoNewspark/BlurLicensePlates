# Create data.yaml for YOLOv7
import yaml

data_yaml = {
    'train': 'C:/data/input/train/images',  # path to train images folder
    'val': 'C:/data/input/valid/images',     # path to validation images folder
    'test': 'C:/data/input/test/images',     # path to test images folder
    
    # Number of classes
    'nc': 1,
    
    # Class names
    'names': ['license_plate']
}

# Save the YAML file
with open('data.yaml', 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)