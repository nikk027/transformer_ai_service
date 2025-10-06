import os
from src.logger.loggerConfig import logger
import yolov9
import warnings
warnings.filterwarnings("ignore")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

global combine_dictionary

combine_dictionary = {
    0: "Road Traffic Text Sign Board",
    1: "Road Sign Single Pole",
    2: "Road Sign Double Pole",
    3: "Road Sign Triple Pole",
    4: "Advertising Sign",
    5: "Single Mast Arm",
    6: "High Mast Arm",
    7: "Monotube",
    8: "Decorative",
    9: "Combo Signal Light",
    10: "Other Luminaire",
    11: "Traffic Signal",
    12: "Traffic Signal Pole",
    13: "Mailbox",
    14: "Reflector",
    15: "Fire Hydrant",
    16: "Support Utility Pole",
    17: "Address Street Name Sign Board",
    18: "Symbol Sign Board",
    19: "Wooden Single Mast Arm",
    20: "Address Number Sign",
    21: "Bollard"
}

# ------------------------------------------------------------------
# Per-class confidence thresholds 
# ------------------------------------------------------------------
class_conf_thresholds = {
    0: 0.17,   # Road Traffic Text Sign Board
    1: 0.257,  # Road Sign Single Pole
    2: 0.268,  # Road Sign Double Pole
    3: 0.422,  # Road Sign Triple Pole
    4: 0.304,  # Advertising Signs
    5: 0.300,  # Single Mast Arm
    6: 0.319,  # High Mast Arm
    7: 0.295,  # Monotube
    8: 0.20,  # Decorative
    9: 0.252,  # Combo Signal Light
    10: 0.295, # Others Luminaire
    11: 0.274, # Traffic Signals
    12: 0.373, # Traffic Signals Pole
    13: 0.34, # Mailboxes
    14: 0.302, # Reflector
    15: 0.275, # Fire Hydrant
    16: 0.239, # Support Utility Pole
    17: 0.22,  # Address Street Name Sign Board
    18: 0.17,  # Symbol Sign Board
    20: 0.15,   # Address Number Sign
    21: 0.30, # Bollards
}

# ------------------------------------------------------------------
# General AI configs
# ------------------------------------------------------------------
default_conf = None  
iou = 0.50
left_crop = 250
upper_crop = 0
right_crop = 7750
lower_crop = 3000

# Classes to include
excludeClasses = []
classes = [x for x in range(0, len(combine_dictionary))]
for i in excludeClasses:
    try:
        classes.remove(i)
    except Exception:
        logger.error(f'Number {i} is not in list')

#device = "cuda" if torch.cuda.is_available() else "cpu"
#Cpu based torch installed in env ,Use Cpu for prediction
device="cpu"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
model_path = os.path.join(project_root, 'model_pt', 'combine.pt')
model_combine = yolov9.load(model_path)

model_combine.to(device)
model_combine.classes = None
model_combine.conf = 0.12  # lowest conf to allow filtering in post-processing
model_combine.iou = iou

DEBUG_MODE = False

# Class for processing images
class ImageProcessor:
    def __init__(self):
        self.model_combine = model_combine
        self.combine_dictionary = combine_dictionary
        self.class_conf_thresholds = class_conf_thresholds
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

    # Combine model prediction
    async def process_combine(self, image_data):
        try:
            self.model_combine.classes = self.classes
            results = self.model_combine(image_data, size=2048)
            if self.DEBUG_MODE:
                await self._save_img(results.plot(conf=False, labels=True, line_width=2))
            flag = None
            logger.info("Combine model image processing completed")
            return self.process_results(results, flag, self.combine_dictionary)
        except Exception as e:
            logger.error(f"Error processing image Combine model: {e}", exc_info=True)
            return None

    # Post-process results with per-class confidence thresholds
    def process_results(self, result, flag, model_dictionary=None):
        try:
            processed_results = []
            preds = result.pred[0]

            if preds is None or len(preds) == 0:
                return processed_results

            for i in range(len(preds)):
                cls = int(preds[i][5].cpu())
                bbox = preds[i][:4].cpu().numpy()
                confi = float(preds[i][4].cpu())
                if cls == 19:
                    threshold = 0.70  # Use threshold of 0.70 for class 19
                else:
                    threshold = self.class_conf_thresholds.get(cls, self.default_conf)
                if confi < threshold:
                    continue
                cl_string = model_dictionary.get(cls, f"class_{cls}")
                masks = result.masks.xy if flag == 1 else None

                out_list = [bbox, cl_string, confi, masks]
                processed_results.append(out_list)

            logger.info("Combine model result processing completed")
            return processed_results
        except Exception as e:
            logger.error(f"Error processing results of Combine model: {e}", exc_info=True)
            return None


    # Save plot images (debug)
    async def _save_img(self, image_data):
        # Code to save image data
        pass
