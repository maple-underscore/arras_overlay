#!/usr/bin/env python3
"""Downscale an image to 320px and run YOLOv8 detection using yolo26n.pt."""

import argparse
import os
import sys
from pathlib import Path

# Force CPU-only mode to avoid CUDA initialization hang
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


def downscale(image_path: str, target_size: int = 320) -> Image.Image:
    """Load and downscale an image so its longest side is target_size. If target_size is 0, don't downscale."""
    img = Image.open(image_path)
    if target_size == 0:
        return img
    ratio = target_size / max(img.size)
    if ratio < 1:
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def analyze(image_path: str, model_path: str = "yolo26n.pt", target_size: int = 320,
            output_path: str | None = None):
    """Downscale the image, run YOLO inference, and save annotated output."""
    img = downscale(image_path, target_size)
    print(f"Image resized to {img.size[0]}x{img.size[1]}")

    model = YOLO(model_path)
    results = model(img)

    # Draw bounding boxes and labels
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  {label}: {conf:.2f}  [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)

            # Draw label background + text
            text = f"{label} {conf:.0%}"
            bbox = draw.textbbox((x1, y1), text, font=font)
            draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1], fill="lime")
            draw.text((x1, y1), text, fill="black", font=font)

    if not any(r.boxes for r in results):
        print("No detections.")

    # Save annotated image
    if output_path is None:
        p = Path(image_path)
        output_path = str(p.with_stem(p.stem + "_detected"))
    img.save(output_path)
    print(f"Annotated image saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO object detection on a downscaled image or directory of images")
    parser.add_argument("input", help="Path to the input image or directory")
    parser.add_argument("--size", type=int, default=320, help="Target downscale size (default: 320)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: path not found: {args.input}")
    
    # Gather all image files to process
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    images_to_process = []
    
    if input_path.is_file():
        if input_path.suffix.lower() in image_extensions:
            images_to_process.append(input_path)
        else:
            sys.exit(f"Error: {args.input} is not a supported image format")
    elif input_path.is_dir():
        # Get all image files in the directory at start time (to avoid processing outputs)
        for file in input_path.iterdir():
            if file.is_file() and file.suffix.lower() in image_extensions:
                images_to_process.append(file)
        if not images_to_process:
            sys.exit(f"Error: no image files found in directory: {args.input}")
        print(f"Found {len(images_to_process)} image(s) to process")
    else:
        sys.exit(f"Error: {args.input} is neither a file nor a directory")
    
    # Models to use
    models = [
        ("nano", "yolo26n.pt", "_1n"),
        ("small", "yolo26s.pt", "_2s"),
        ("medium", "yolo26m.pt", "_3m")
    ]
    
    # Process each image with all models
    for image_path in images_to_process:
        print(f"\n{'='*60}")
        print(f"Processing: {image_path.name}")
        print(f"{'='*60}")
        
        for model_name, model_path, suffix in models:
            if not Path(model_path).exists():
                print(f"Warning: model {model_path} not found, skipping {model_name}")
                continue
                
            print(f"\nAnalyzing with {model_name} model ({model_path})...")
            
            # Construct output filename
            stem = image_path.stem
            ext = image_path.suffix
            output_filename = f"{stem}{suffix}{ext}"
            output_path = image_path.parent / output_filename
            
            try:
                analyze(str(image_path), model_path, args.size, str(output_path))
            except Exception as e:
                print(f"Error processing {image_path.name} with {model_name}: {e}")
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")
