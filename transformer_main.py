import os
import json
import torch
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import random

# ---------------------------
# Setup
# ---------------------------
processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Class prompts
# texts = [["safety bollard","yellow bollard","Yellow Bollards","signal cabinates","Street power pedestal ","power-electrical-transformer",
#           'public post boxes',"curbside mailboxes","communal mailboxes","Cluster Mailboxes","Decorative and novelty mailboxes",'Street Mailbox','traffic signal control cabinet','electrical power pedestal',
#           'fixed steel bollard',"Street Bollards"   
#           'yellow fire hydrants','green fire hydrant','blue fire hydrant','Red Fire Hydrant','Transformer box',
#           "Street Light Luminaires",'Combo Signal Pole','Wire Support Utility Pole'
#           'Wrong Side Traffic Sign Board','Symbol Traffic Sign Board','Street Name Sign Board','Stop Sign Traffic Board','speed limit sign board','handicap sign board',
#           'traffic signals',"street sign pole","sign board pole","metal pole supporting a road sign","traffic sign support post","vertical pole with road sign"]]
# texts = [["Pole Transformer","Light Pole transformer",'Wire Support Utility Pole',"pole-mounted transformer", "insulator", "crossarm", "fuse cutout", "lightning arrester", "pole-mounted"]]
# texts = [
#     [
#         "road traffic sign board with any text",
#         "road traffic sign board with street name",
#         "road traffic sign board showing traffic symbols",
#         "numbered street sign board",
#         "advertising sign board on the road",
#         "Wire Support Utility Pole",
#         "traffic signal mounted on a pole",
#         "street luminaire or lamp post",
#         "traffic signal on road",
#         "Collection Box",
#         "Post Box",
#         "roadside reflector for caution",
#         "yellow fire hydrants",
#         "green fire hydrant",
#         "blue fire hydrant",
#         "Red Fire Hydrant",
#         "Electric Transformer box",
#         "White bollard",
#         "yellow bollard",
#         "Yellow Bollards",
#         "White bollards",
#         "Wrong Side Traffic Sign Board",
#         "Symbol Traffic Sign Board",
#         "Street Name Sign Board",
#         "Stop Sign Traffic Board",
#         "speed limit sign board",
#         "handicap sign board",
#         "Traffic Sign Boards",
#         "metal pole supporting a road sign board",
#         "traffic sign support post",
#         "Street Mailbox",
#     ]
# ]

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
        label_colors[label] = tuple([random.randint(0, 255) for _ in range(3)])
    return label_colors[label]

# ---------------------------
# Font for labels (bigger text)
# ---------------------------
try:
    font = ImageFont.truetype("arial.ttf", 24)  # Windows usually has Arial
except:
    font = ImageFont.load_default()

# ---------------------------
# Process images with timing + progress bar
# ---------------------------
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

start_all = time.time()  # total timer

for file_name in tqdm(image_files, desc="Processing images"):
    file_path = os.path.join(image_folder, file_name)

    try:
        start_img = time.time()  # per-image timer

        image = Image.open(file_path).convert("RGB")

        # Preprocess inputs
        inputs = processor(text=texts, images=image, return_tensors="pt").to(device)

        # Run model
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

        # Draw detections on ORIGINAL image
        visualized_image = image.copy()
        draw = ImageDraw.Draw(visualized_image)

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [round(coord, 2) for coord in box.tolist()]

            # Draw thick bounding box
            color = get_label_color(text[label])
            draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=6)

            # Draw label text with bigger font
            label_text = f"{text[label]}: {round(score.item(), 2)}"
            draw.text((x1, y1 - 25), label_text, fill=color, font=font)

            # Save detection in COCO JSON format
            coco_results.append({
                "image_id": file_name,
                "category_id": int(label.item()),  # class index
                "category_name": text[label],     # class name
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO: [x,y,width,height]
                "score": round(score.item(), 3)
            })

        # Save visualized image
        save_path = os.path.join(output_folder, f"det_{file_name}")
        visualized_image.save(save_path)

        # Print per-image timing
        elapsed_img = time.time() - start_img
        print(f"⏱ Processed {file_name} in {elapsed_img:.2f} sec")

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# ---------------------------
# Save COCO JSON + total timing
# ---------------------------
with open(json_output_path, "w") as f:
    json.dump(coco_results, f, indent=4)

elapsed_all = time.time() - start_all
print(f"\n✅ Processing complete. Results saved in:\n- {output_folder}\n- {json_output_path}")
print(f"⏳ Total time for {len(image_files)} images: {elapsed_all:.2f} sec "
      f"({elapsed_all/len(image_files):.2f} sec/image)")



################################_______________________________________SAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMMMMMMMMMMMMMMMMMMMMMMMMM-----------------------------------------------------
