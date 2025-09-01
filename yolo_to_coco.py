# yolo_to_coco.py
import os
import json
import cv2
from tqdm import tqdm

def yolo_to_coco(yolo_dir, output_json, split='train'):
    """
    Convert YOLO format annotations to COCO JSON format
    """
    # Define paths
    images_dir = os.path.join(yolo_dir, f'VisDrone2019-DET-{split}', 'images')
    labels_dir = os.path.join(yolo_dir, f'VisDrone2019-DET-{split}', 'labels')
    
    # COCO dataset structure
    coco_dataset = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    # Define categories (your 9 classes)
    categories = [
        {"id": 0, "name": "pedestrian", "supercategory": "person"},
        {"id": 1, "name": "people", "supercategory": "person"},
        {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
        {"id": 3, "name": "car", "supercategory": "vehicle"},
        {"id": 4, "name": "van", "supercategory": "vehicle"},
        {"id": 5, "name": "truck", "supercategory": "vehicle"},
        {"id": 6, "name": "tricycle", "supercategory": "vehicle"},
        {"id": 7, "name": "awning-tricycle", "supercategory": "vehicle"},
        {"id": 8, "name": "bus", "supercategory": "vehicle"},
        {"id": 9, "name": "motor", "supercategory": "vehicle"}
    ]
    coco_dataset["categories"] = categories
    
    # Process each image
    annotation_id = 0
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_id, image_file in enumerate(tqdm(image_files)):
        # Add image info
        image_path = os.path.join(images_dir, image_file)
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        coco_dataset["images"].append({
            "id": image_id,
            "file_name": image_file,
            "width": width,
            "height": height
        })
        
        # Process annotations
        label_file = image_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(labels_dir, label_file)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                data = line.strip().split()
                if len(data) == 5:
                    class_id, x_center, y_center, bbox_width, bbox_height = map(float, data)
                    
                    # Convert YOLO to COCO format (normalized → absolute)
                    x_center_abs = x_center * width
                    y_center_abs = y_center * height
                    bbox_width_abs = bbox_width * width
                    bbox_height_abs = bbox_height * height
                    
                    # Convert to COCO bbox format [x, y, width, height]
                    x_min = x_center_abs - (bbox_width_abs / 2)
                    y_min = y_center_abs - (bbox_height_abs / 2)
                    
                    coco_dataset["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(class_id),
                        "bbox": [x_min, y_min, bbox_width_abs, bbox_height_abs],
                        "area": bbox_width_abs * bbox_height_abs,
                        "iscrowd": 0
                    })
                    annotation_id += 1
    
    # Save COCO JSON
    with open(output_json, 'w') as f:
        json.dump(coco_dataset, f, indent=2)
    
    print(f"Converted {split} set: {len(coco_dataset['images'])} images, {len(coco_dataset['annotations'])} annotations")

# Convert all splits
yolo_base_dir = "D:/datasets/visdrone"

# Create annotations directory
annotations_dir = os.path.join(yolo_base_dir, "annotations")
os.makedirs(annotations_dir, exist_ok=True)

# Convert each split
for split in ['train', 'val', 'test-dev']:
    output_json = os.path.join(annotations_dir, f"instances_{split}.json")
    yolo_to_coco(yolo_base_dir, output_json, split)