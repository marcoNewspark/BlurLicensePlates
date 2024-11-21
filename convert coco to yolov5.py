import json
import os
from pathlib import Path
import cv2

def coco_to_yolo(json_path, output_dir):
    """
    Convert COCO format annotations to YOLO format.
    
    Args:
        json_path: Path to COCO json file
        output_dir: Directory to save YOLO format txt files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read COCO JSON file
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create lookup dictionaries
    image_dict = {img['id']: img for img in coco_data['images']}
    
    # Process annotations
    for ann in coco_data['annotations']:
        # Get image details
        img_info = image_dict[ann['image_id']]
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Get bbox coordinates
        x, y, w, h = ann['bbox']
        
        # Convert to YOLO format
        # YOLO format is: <class> <x_center> <y_center> <width> <height>
        # where all values are normalized between 0 and 1
        x_center = (x + w/2) / img_width
        y_center = (y + h/2) / img_height
        norm_width = w / img_width
        norm_height = h / img_height
        
        # Class ID (for license plates, usually just 0 as we have one class)
        class_id = 0
        
        # Create YOLO format line
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
        
        # Write to file
        output_filename = os.path.splitext(img_info['file_name'])[0] + '.txt'
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'a') as f:
            f.write(yolo_line + '\n')

def verify_conversion(image_dir, label_dir):
    """
    Utility function to visualize the converted annotations
    """
    import cv2
    import numpy as np
    
    def draw_yolo_box(img, x_center, y_center, w, h):
        img_h, img_w = img.shape[:2]
        # Convert normalized coordinates back to pixel coordinates
        x = int((x_center - w/2) * img_w)
        y = int((y_center - h/2) * img_h)
        width = int(w * img_w)
        height = int(h * img_h)
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
        return img

    # Process each image and its corresponding label
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
            
        image_file = os.path.splitext(label_file)[0] + '.jpg'
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_file} not found")
            continue
            
        # Read image and labels
        img = cv2.imread(image_path)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        # Draw each box
        for line in lines:
            class_id, x_center, y_center, w, h = map(float, line.strip().split())
            img = draw_yolo_box(img, x_center, y_center, w, h)
            
        # Display image
        cv2.imshow('Verification', img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    config = {
        'coco_json': 'C:/data/input/test/_annotations.coco.json',
        'output_dir': 'C:/data/input/test',
        'image_dir': 'C:/data/input/test'  # For verification
    }
    
    # Convert annotations
    coco_to_yolo(config['coco_json'], config['output_dir'])
    
    # Optional: Verify conversions
    print("Would you like to verify the conversions? (y/n)")
    if input().lower() == 'y':
        verify_conversion(config['image_dir'], config['output_dir'])