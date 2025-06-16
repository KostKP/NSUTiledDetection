import argparse
import cv2
import numpy as np
import os
from pathlib import Path
import torch

config = {
    "tile_size": 640,
    "tile_overlap": 48,
    "model_weights" : "./yolov5/runs/train/human_detect_heridal/weights/best.pt",
    "conf_threshold" : 0.5,
    "output_dir" : "./outputs",
    "image_extensions": ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
}

def split_image(img, tile_size=640, overlap=48):
    h, w = img.shape[:2]
    
    x_tiles = (w - overlap) // (tile_size - overlap) + 1
    y_tiles = (h - overlap) // (tile_size - overlap) + 1
    
    new_w = (tile_size - overlap) * x_tiles + overlap
    new_h = (tile_size - overlap) * y_tiles + overlap
    pad_w = max(new_w - w, 0)
    pad_h = max(new_h - h, 0)
    
    img_padded = cv2.copyMakeBorder(img, pad_h//2, pad_h - pad_h//2, pad_w//2, pad_w - pad_w//2, cv2.BORDER_CONSTANT)
    
    tiles = []
    coords = []
    
    for y in range(y_tiles):
        for x in range(x_tiles):
            x0 = x*(tile_size - overlap)
            y0 = y*(tile_size - overlap)
            x1 = x0 + tile_size
            y1 = y0 + tile_size
            
            tile = img_padded[y0:y1, x0:x1]
            tiles.append(tile)
            coords.append((x0 - pad_w//2, y0 - pad_h//2, x1 - pad_w//2, y1 - pad_h//2))
    
    return tiles, coords, (pad_w, pad_h)

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1[1:5]
    x2_min, y2_min, x2_max, y2_max = box2[1:5]
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    return inter_area / (area1 + area2 - inter_area)

def cover_ratio(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1[1:5]
    x2_min, y2_min, x2_max, y2_max = box2[1:5]

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    return inter_area / area2 if area2 > 0 else 0.0

def nms(boxes, iou_threshold=0.5, cover_threshold=0.9):
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda x: x[5], reverse=True)
    keep = []

    while sorted_boxes:
        current = sorted_boxes.pop(0)
        keep.append(current)
        to_remove = []

        for i, box in enumerate(sorted_boxes):
            if current[0] != box[0]:
                continue

            iou = calculate_iou(current, box)
            cover = cover_ratio(current, box)

            if iou > iou_threshold or cover > cover_threshold:
                to_remove.append(i)

        for i in reversed(to_remove):
            sorted_boxes.pop(i)

    return keep

def process_image(image_path, model, output_dir, conf_threshold, output_type):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return

    h, w = img.shape[:2]
    tiles, coords, (pad_w, pad_h) = split_image(img, config["tile_size"], config["tile_overlap"])
    
    all_boxes = []
    for tile, coord in zip(tiles, coords):
        results = model(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        preds = results.xyxyn[0].cpu().numpy()
        
        for pred in preds:
            x1, y1, x2, y2, conf, cls = pred
            if conf < conf_threshold:
                continue
            
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            tile_size = config["tile_size"]
            x0_tile, y0_tile, x1_tile, y1_tile = coord
            
            x_center_abs = x0_tile + x_center * tile_size
            y_center_abs = y0_tile + y_center * tile_size
            width_abs = width * tile_size
            height_abs = height * tile_size
            
            xmin = x_center_abs - width_abs / 2
            ymin = y_center_abs - height_abs / 2
            xmax = x_center_abs + width_abs / 2
            ymax = y_center_abs + height_abs / 2
            
            all_boxes.append((int(cls), xmin, ymin, xmax, ymax, float(conf)))
    
    filtered_boxes = nms(all_boxes)
    
    final_boxes = []
    for box in filtered_boxes:
        cls, xmin, ymin, xmax, ymax, conf = box
        
        xmin_adj = max(xmin, 0)
        ymin_adj = max(ymin, 0)
        xmax_adj = min(xmax, w)
        ymax_adj = min(ymax, h)
        
        if xmax_adj > xmin_adj and ymax_adj > ymin_adj: final_boxes.append((cls, xmin_adj, ymin_adj, xmax_adj, ymax_adj, conf))
    
    input_filename = image_path.stem
    output_img_path = output_dir / f"{input_filename}.jpg"
    output_txt_path = output_dir / f"{input_filename}.txt"
    
    if output_type in ["both", "image"]:
        output_img = img.copy()
        for box in final_boxes:
            cls, xmin, ymin, xmax, ymax, conf = box
            cv2.rectangle(output_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            label = f"{model.names[int(cls)]}: {conf:.2f}"
            cv2.putText(output_img, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(str(output_img_path), output_img)
    
    if output_type in ["both", "text"]:
        with open(output_txt_path, 'w') as f:
            for box in final_boxes:
                f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]:.4f}\n")
    
    print(f"Processed: {image_path.name}")

def main():
    parser = argparse.ArgumentParser(description='Process images and predict bounding boxes.')
    parser.add_argument('input_path', type=str, help='Path to input image or folder.')
    parser.add_argument('--output_dir', type=str, default=config["output_dir"], help='Output directory.')
    parser.add_argument('--model_weights', type=str, default=config["model_weights"], help='Path to model weights.')
    parser.add_argument('--conf_threshold', type=float, default=config["conf_threshold"], help='Confidence threshold.')
    parser.add_argument('--output_type', type=str, default='both', choices=['image', 'text', 'both'], help='Output type: image, text, or both.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.model_weights).to(device)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        if input_path.suffix.lower() in config["image_extensions"]:
            process_image(input_path, model, output_dir, args.conf_threshold, args.output_type)
        else:
            print(f"Unsupported file format: {input_path}")
    
    elif input_path.is_dir():
        image_files = []
        for ext in config["image_extensions"]:
            image_files.extend(list(input_path.glob(f'*{ext}')))
            image_files.extend(list(input_path.glob(f'*{ext.upper()}')))

        if not image_files:
            print(f"No valid images found in: {input_path}")
            return
            
        print(f"Found {len(image_files)} images to process")
        for image_path in image_files:
            process_image(image_path, model, output_dir, args.conf_threshold, args.output_type)
    else:
        print(f"Invalid input path: {input_path}")

if __name__ == "__main__":
    main()
