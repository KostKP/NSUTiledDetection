# Human Detection with YOLOv5 on Tiled Images

This project detects humans in large images using a tiled approach with YOLOv5. It handles large images by splitting them into overlapping tiles, running detection on each tile, and merging results using Non-Maximum Suppression (NMS).

## Features

- Split large images into tiles with overlap
- Perform detection using YOLOv5
- Merge bounding boxes with custom NMS and cover ratio filtering
- Save output as image, text, or both

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Detect on a Single Image

```bash
python detect.py path/to/image.jpg --model_weights path/to/best.pt --output_type both
```

## Detect on All Images in a Folder

```bash
python detect.py path/to/image_folder --model_weights path/to/best.pt --output_type both
```

## Arguments

| Argument           | Description                                                 |
| ------------------ | ----------------------------------------------------------- |
| `input_path`       | Path to an image file or a folder containing images         |
| `--output_dir`     | Output directory to save results (default: `./outputs`)     |
| `--model_weights`  | Path to the YOLOv5 model weights (`.pt` file)               |
| `--conf_threshold` | Confidence threshold to filter detections (default: `0.5`)  |
| `--output_type`    | Output format: `image`, `text`, or `both` (default: `both`) |

## Outputs

Each processed image generates:
- .jpg image with bounding boxes (if --output_type includes image)
- .txt file with box coordinates and confidence scores (if --output_type includes text)
