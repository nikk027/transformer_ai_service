import os
import json
import torch
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import random
import cv2

# SAM imports - install with: pip install segment-anything
try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
    print("✅ SAM imported successfully")
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️  SAM not installed. Install with: pip install segment-anything")
    print("   Falling back to bounding box mode")

# ---------------------------
# Setup OWL-v2 Model
# ---------------------------
processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"🔥 Using device: {device}")

# ---------------------------
# Setup SAM Model
# ---------------------------
sam_predictor = None
if SAM_AVAILABLE:
    try:
        # Download SAM checkpoint if not exists
        sam_checkpoint = "sam_vit_h_4b8939.pth"  # You need to download this
        model_type = "vit_h"
        
        if os.path.exists(sam_checkpoint):
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            sam_predictor = SamPredictor(sam)
            print("✅ SAM model loaded successfully")
        else:
            print(f"⚠️  SAM checkpoint '{sam_checkpoint}' not found")
            print("   Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            SAM_AVAILABLE = False
    except Exception as e:
        print(f"❌ Error loading SAM: {e}")
        SAM_AVAILABLE = False

# Class prompts
texts = [["Pothole", "Potholes", "cracks on road", "crack on road"]]

# Input/output folders
image_folder = r"\\172.25.10.197\s2m\Data-Road_Distresses\Sample\Input Video Files\Batch078B\Batch078\Pavement\LCMS3D\FISPavement\20241206.140614"
output_folder = r"X:\nikhil\output_val\raod2"
os.makedirs(output_folder, exist_ok=True)

# JSON output file
json_output_path = os.path.join(output_folder, "detections_coco.json")
coco_results = []

# ---------------------------
# Assign unique random colors for each label
# ---------------------------
label_colors = {}
def get_label_color(label):
    if label not in label_colors:
        label_colors[label] = tuple([random.randint(50, 255) for _ in range(3)])
    return label_colors[label]

# ---------------------------
# Font for labels (bigger text)
# ---------------------------
try:
    font = ImageFont.truetype("arial.ttf", 24)  # Windows usually has Arial
except:
    font = ImageFont.load_default()

# ---------------------------
# Function to apply colored mask overlay
# ---------------------------
def apply_mask_overlay(image, mask, color, alpha=0.5):
    """Apply colored semi-transparent mask overlay to image"""
    overlay = image.copy()
    overlay_array = np.array(overlay)
    
    # Create colored mask
    colored_mask = np.zeros_like(overlay_array)
    colored_mask[mask] = color
    
    # Blend with original image
    mask_area = mask.astype(np.uint8) * 255
    mask_3channel = np.stack([mask_area] * 3, axis=-1)
    
    # Apply alpha blending
    blended = cv2.addWeighted(overlay_array, 1-alpha, colored_mask, alpha, 0)
    result = np.where(mask_3channel > 0, blended, overlay_array)
    
    return Image.fromarray(result.astype(np.uint8))

# ---------------------------
# Function to get segmentation mask from bounding box
# ---------------------------
def get_segmentation_mask(image, bbox):
    """Convert bounding box to segmentation mask using SAM"""
    if not SAM_AVAILABLE or sam_predictor is None:
        return None
    
    try:
        # Convert PIL to numpy for SAM
        image_array = np.array(image)
        sam_predictor.set_image(image_array)
        
        # Convert bbox to SAM format [x_min, y_min, x_max, y_max]
        input_box = np.array(bbox)
        
        # Get mask from SAM
        masks, scores, logits = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        
        # Return the best mask
        return masks[0] if len(masks) > 0 else None
        
    except Exception as e:
        print(f"❌ SAM prediction error: {e}")
        return None

# ---------------------------
# Process images with timing + progress bar
# ---------------------------
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

start_all = time.time()  # total timer
mode = "SEGMENTATION" if SAM_AVAILABLE and sam_predictor else "BOUNDING BOX"
print(f"🎯 Processing {len(image_files)} images in {mode} mode")

for file_name in tqdm(image_files, desc=f"Processing images ({mode})"):
    file_path = os.path.join(image_folder, file_name)

    try:
        start_img = time.time()  # per-image timer

        image = Image.open(file_path).convert("RGB")

        # Preprocess inputs for OWL-v2
        inputs = processor(text=texts, images=image, return_tensors="pt").to(device)

        # Run OWL-v2 model
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process outputs with ORIGINAL image size
        target_sizes = torch.tensor([image.size[::-1]]).to(device)  # (height, width)
        results = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.10
        )

        i = 0
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        # Start with original image for visualization
        visualized_image = image.copy()
        draw = ImageDraw.Draw(visualized_image)

        # Process each detection
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [round(coord, 2) for coord in box.tolist()]
            color = get_label_color(text[label])
            
            if SAM_AVAILABLE and sam_predictor:
                # SAM SEGMENTATION MODE
                mask = get_segmentation_mask(image, [x1, y1, x2, y2])
                
                if mask is not None:
                    # Apply mask overlay to image
                    visualized_image = apply_mask_overlay(visualized_image, mask, color, alpha=0.4)
                    
                    # Draw contours for better visibility
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Convert back to PIL for contour drawing
                    img_array = np.array(visualized_image)
                    cv2.drawContours(img_array, contours, -1, color, 3)
                    visualized_image = Image.fromarray(img_array)
                    
                    # Calculate mask area for more accurate data
                    mask_area = np.sum(mask)
                    
                    # Save detection with mask info in COCO JSON format
                    coco_results.append({
                        "image_id": file_name,
                        "category_id": int(label.item()),
                        "category_name": text[label],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO: [x,y,width,height]
                        "score": round(score.item(), 3),
                        "segmentation_area": int(mask_area),
                        "has_mask": True
                    })
                else:
                    # Fallback to bounding box if SAM fails
                    draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=6)
                    coco_results.append({
                        "image_id": file_name,
                        "category_id": int(label.item()),
                        "category_name": text[label],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": round(score.item(), 3),
                        "has_mask": False
                    })
            else:
                # BOUNDING BOX MODE (fallback)
                draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=6)
                coco_results.append({
                    "image_id": file_name,
                    "category_id": int(label.item()),
                    "category_name": text[label],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": round(score.item(), 3),
                    "has_mask": False
                })

            # Draw label text
            label_text = f"{text[label]}: {round(score.item(), 2)}"
            draw.text((x1, y1 - 25), label_text, fill=color, font=font)

        # Save visualized image
        save_path = os.path.join(output_folder, f"det_{file_name}")
        visualized_image.save(save_path)

        # Print per-image timing
        elapsed_img = time.time() - start_img
        print(f"⏱ Processed {file_name} in {elapsed_img:.2f} sec ({mode})")

    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")

# ---------------------------
# Save COCO JSON + total timing
# ---------------------------
with open(json_output_path, "w") as f:
    json.dump(coco_results, f, indent=4)

elapsed_all = time.time() - start_all
print(f"\n✅ Processing complete in {mode} mode. Results saved in:\n- {output_folder}\n- {json_output_path}")
print(f"⏳ Total time for {len(image_files)} images: {elapsed_all:.2f} sec "
      f"({elapsed_all/len(image_files):.2f} sec/image)")

# Print summary
total_detections = len(coco_results)
mask_detections = sum(1 for r in coco_results if r.get("has_mask", False))
print(f"📊 Summary: {total_detections} total detections, {mask_detections} with segmentation masks")

################################_______________________________________SAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMMMMMMMMMMMMMMMMMMMMMMMMM-----------------------------------------------------