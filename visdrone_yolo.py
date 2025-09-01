# visdrone_yolo.py
import os
import cv2

def convert_visdrone_to_yolo(visdrone_split_path):
    """
    Converts a VisDrone split to YOLO format.
    
    Args:
        visdrone_split_path (str): Full path to the split folder (e.g., 'D:/datasets/visdrone/VisDrone2019-DET-train')
    """
    
    annotations_dir = os.path.join(visdrone_split_path, 'annotations')
    images_dir = os.path.join(visdrone_split_path, 'images')
    yolo_labels_dir = os.path.join(visdrone_split_path, 'labels')  # New folder for YOLO labels

    # Create the output directory for YOLO labels
    os.makedirs(yolo_labels_dir, exist_ok=True)
    print(f"Processing: {visdrone_split_path}")
    print(f"Saving YOLO labels to: {yolo_labels_dir}")

    # VisDrone category mapping
    category_map = {
        0: 0,   # "ignored regions" -> class 0 (filtered out)
        1: 0,   # "pedestrian"
        2: 0,   # "people" (merged)
        3: 1,   # "bicycle"
        4: 2,   # "car"
        5: 3,   # "van"
        6: 4,   # "truck"
        7: 5,   # "tricycle"
        8: 6,   # "awning-tricycle"
        9: 7,   # "bus"
        10: 8,  # "motor"
    }

    # Loop through all annotation files in the split
    ann_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
    
    for i, ann_file in enumerate(ann_files):
        if i % 500 == 0:
            print(f"Processing {i}/{len(ann_files)}...")

        ann_path = os.path.join(annotations_dir, ann_file)
        img_name = ann_file.replace('.txt', '.jpg')
        img_path = os.path.join(images_dir, img_name)

        # Check if the corresponding image exists
        if not os.path.exists(img_path):
            continue

        # Get image dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w, _ = img.shape

        yolo_ann_lines = []
        with open(ann_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = line.strip().split(',')
            if len(data) < 6:
                continue

            # Extract values
            x_abs = int(data[0])
            y_abs = int(data[1])
            w_abs = int(data[2])
            h_abs = int(data[3])
            category_id = int(data[5])
            score = float(data[4])

            # Filter out low-confidence and ignored regions
            if score < 0.1 or category_id == 0:
                continue

            # Map category ID
            mapped_class_id = category_map.get(category_id, -1)
            if mapped_class_id == -1:
                continue

            # Convert to YOLO format
            x_center = (x_abs + w_abs / 2) / img_w
            y_center = (y_abs + h_abs / 2) / img_h
            w_norm = w_abs / img_w
            h_norm = h_abs / img_h

            # Check if coordinates are valid
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < w_norm <= 1 and 0 < h_norm <= 1):
                continue

            yolo_line = f"{mapped_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
            yolo_ann_lines.append(yolo_line)

        # Write the converted annotations
        output_file_path = os.path.join(yolo_labels_dir, ann_file)
        with open(output_file_path, 'w') as out_f:
            out_f.writelines(yolo_ann_lines)

    print(f"Finished processing {visdrone_split_path}.")

# --- RUN THE CONVERSION FOR TRAIN AND VAL SPLITS ---
base_path = "D:/datasets/visdrone"

# Convert all necessary splits
train_split_path = os.path.join(base_path, "VisDrone2019-DET-train")
val_split_path = os.path.join(base_path, "VisDrone2019-DET-val")

convert_visdrone_to_yolo(visdrone_split_path=train_split_path)
print("\n" + "="*50 + "\n")
convert_visdrone_to_yolo(visdrone_split_path=val_split_path)

print("All conversions complete!")