import os
import cv2
import glob
import uuid
import albumentations as A

# --------- CONFIG ----------
DATA_DIR = "dataset"        # parent folder containing 'images/' and 'labels/'
AUG_PER_IMAGE = 5           # how many augmentations per original image
IMG_DIR = os.path.join(DATA_DIR, "images")
LBL_DIR = os.path.join(DATA_DIR, "labels")

# --------- HELPER FUNCTIONS ----------
def read_label_file(path):
    """Reads YOLO labels: class x_center y_center w h"""
    bboxes = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # Take only first 5 values in case of malformed files
            cls, x, y, w, h = parts[0], parts[1], parts[2], parts[3], parts[4]
            # Clip coordinates to valid range [0, 1]
            x, y, w, h = max(0.0, min(1.0, float(x))), max(0.0, min(1.0, float(y))), max(0.0, min(1.0, float(w))), max(0.0, min(1.0, float(h)))
            bboxes.append([int(float(cls)), x, y, w, h])
    return bboxes

def write_label_file(path, bboxes):
    """Writes YOLO labels back"""
    with open(path, "w") as f:
        for bbox in bboxes:
            cls, x, y, w, h = bbox
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def clip_bbox(bbox):
    """Clip bounding box coordinates to [0,1]"""
    return [max(0.0, min(1.0, v)) for v in bbox]

# --------- DEFINE AUGMENTATIONS ----------
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.7),
    A.HueSaturationValue(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

# --------- PROCESS ALL IMAGES ----------
image_paths = glob.glob(os.path.join(IMG_DIR, "*.*"))

for img_path in image_paths:
    filename = os.path.basename(img_path).split(".")[0]
    label_path = os.path.join(LBL_DIR, f"{filename}.txt")
    if not os.path.exists(label_path):
        continue

    bboxes = read_label_file(label_path)
    if not bboxes:
        continue

    # Separate class IDs and bbox coords
    category_ids = [b[0] for b in bboxes]
    bboxes_only = [[max(0.0, min(1.0, b[1])), max(0.0, min(1.0, b[2])), max(0.0, min(1.0, b[3])), max(0.0, min(1.0, b[4]))] for b in bboxes]  # Ensure all values are [0, 1]

    # Validate bboxes before augmentation
    try:
        img = cv2.imread(img_path)
        
        for i in range(AUG_PER_IMAGE):
            augmented = transform(image=img, bboxes=bboxes_only, category_ids=category_ids)
            aug_img = augmented["image"]
            aug_bboxes = [[cls, *clip_bbox(b)] for cls, b in zip(augmented["category_ids"], augmented["bboxes"])]

            # Unique filenames
            aug_filename = f"{filename}_aug{i+1}_{uuid.uuid4().hex[:6]}"
            cv2.imwrite(os.path.join(IMG_DIR, f"{aug_filename}.png"), aug_img)
            write_label_file(os.path.join(LBL_DIR, f"{aug_filename}.txt"), aug_bboxes)

        print(f"Augmented {filename} x{AUG_PER_IMAGE}")
    except ValueError as e:
        print(f"Skipping {filename}: {e}")
