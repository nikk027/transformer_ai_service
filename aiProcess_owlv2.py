import os
import warnings
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torch
from PIL import Image
import numpy as np
from src.logger.loggerConfig import logger

warnings.filterwarnings("ignore")

# Prompts for OWLv2 (copied from transformer_main.py)
class_prompts = [[
    "safety bollard", "yellow bollard",
    "Yellow Bollards", "signal cabinates", "Street power pedestal ", "power-electrical-transformer",
    'public post boxes', "curbside mailboxes", "communal mailboxes", "Cluster Mailboxes", "Decorative and novelty mailboxes", 'Street Mailbox', 'traffic signal control cabinet', 'electrical power pedestal',
    'fixed steel bollard', "Street Bollards",
    'yellow fire hydrants', 'green fire hydrant', 'blue fire hydrant', 'Red Fire Hydrant', 'Transformer box',
    "Street Light Luminaires", 'Combo Signal Pole', 'Wire Support Utility Pole',
    'Wrong Side Traffic Sign Board', 'Symbol Traffic Sign Board', 'Street Name Sign Board', 'Stop Sign Traffic Board', 'speed limit sign board', 'handicap sign board',
    'traffic signals', "street sign pole", "sign board pole", "metal pole supporting a road sign", "traffic sign support post", "vertical pole with road sign"
]]

# General AI configs
default_conf = 0.35  # Default confidence threshold
iou = 0.50
left_crop = 250
upper_crop = 0
right_crop = 7750
lower_crop = 3000

# Classes to include
excludeClasses = []
classes = [x for x in range(0, len(class_prompts[0]))]
for i in excludeClasses:
    try:
        classes.remove(i)
    except Exception:
        logger.error(f'Number {i} is not in list')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load OWLv2 model and processor
processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble")
model.to(device)

DEBUG_MODE = False


# Class for processing images (same name as aiProcess.py)
class ImageProcessor:
    def __init__(self):
        self.processor = processor
        self.model = model
        self.class_prompts = class_prompts
        self.default_conf = default_conf
        self.iou = iou
        self.classes = classes
        self.DEBUG_MODE = DEBUG_MODE


    # Pre-process image and return results
    async def process_image(self, image_data):
        image_crop = (left_crop, upper_crop, right_crop, lower_crop)
        image_data = image_data.crop(image_crop)
        res_combine = await self.process_combine(image_data)
        if res_combine is None:
            return []
        combined_results = res_combine
        return combined_results, left_crop


    # Combine model prediction using OWLv2
    async def process_combine(self, image_data):
        try:
            inputs = self.processor(text=self.class_prompts, images=image_data, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            target_sizes = torch.tensor([image_data.size[::-1]]).to(device)  # (height, width)
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.default_conf
            )
            logger.info("OWLv2 model image processing completed")
            return self.process_results(results[0], None)
        except Exception as e:
            logger.error(f"Error processing image with OWLv2 model: {e}", exc_info=True)
            return None


    # Post-process results (output: [bbox, cl_string, confi, None])
    def process_results(self, result, flag, model_dictionary=None):
        try:
            processed_results = []
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["labels"]
            if boxes is None or len(boxes) == 0:
                return processed_results
            for i in range(len(boxes)):
                cls = int(labels[i])
                bbox = [float(coord) for coord in boxes[i].tolist()]
                confi = float(scores[i])
                if confi < self.default_conf:
                    continue
                cl_string = self.class_prompts[0][cls] if cls < len(self.class_prompts[0]) else f"class_{cls}"
                masks = None
                out_list = [bbox, cl_string, confi, masks]
                processed_results.append(out_list)
            logger.info("OWLv2 model result processing completed")
            return processed_results
        except Exception as e:
            logger.error(f"Error processing results of OWLv2 model: {e}", exc_info=True)
            return None


    # Save plot images (debug)
    async def _save_img(self, image_data):
        # Code to save image data
        pass